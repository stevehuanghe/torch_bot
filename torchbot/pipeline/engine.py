

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
from pathlib import Path
import pickle as pk
from tqdm import tqdm
from collections import defaultdict
from pprint import pprint
import time
import mlflow
import numpy as np
import warnings
import subprocess as sbp
import random
import shutil
import time
import warnings
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


from .experiment import Experiment
from .py_logger import Logger
from .utils import is_master, get_sampler
from torchbot.utils.misc import clip_grad_norm, to_device, to_item, accuracy
from torchbot.pyutils.misc import AverageMeter, AverageMonitor

class TorchEngine(Experiment):

    def start(self):
        if self.args.cudnn_benchmark:
            cudnn.benchmark = True

        if self.args.seed is not None:
            random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')

        if self.args.gpu is not None:
            warnings.warn('You have chosen a specific GPU. This will completely '
                          'disable data parallelism.')

        if self.args.dist_url == "env://" and self.args.world_size == -1:
            self.args.world_size = int(os.environ["WORLD_SIZE"])

        self.args.distributed = self.args.world_size > 1 or self.args.multiprocessing_distributed

        args = self.args
        args.logfile = self.logfile
        args.paths = self.paths
        ngpus_per_node = torch.cuda.device_count()
        args.ngpus_per_node = ngpus_per_node
        self.args = args
        pprint(vars(self.args))

        if not self.use_mlflow:
            print("not using mlflow")
            self.run()
            return

        
        mlflow.set_tracking_uri(self.mlflow_server)
        experiment_id = mlflow.set_experiment(str(self.mlflow_dir))
        with mlflow.start_run(experiment_id=experiment_id, run_name=self.run_name):
            mlflow.log_param("base_dir", str(self.base_dir))
            mlflow.log_artifact(str(self.param_file))
            for k, v in sorted(vars(self.args).items()):
                mlflow.log_param(k, v)
            # start the program
            self.run()

    def run(self):
        if self.args.multiprocessing_distributed:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            self.args.world_size = self.args.ngpus_per_node * self.args.world_size
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function
            mp.spawn(self.main_worker, nprocs=self.args.ngpus_per_node, args=(self.args.ngpus_per_node, self.args))
        else:
            # Simply call main_worker function
            self.main_worker(self.args.gpu, self.args.ngpus_per_node, self.args)

    def main_worker(self, gpu, ngpus_per_node, args):
        args.gpu = gpu
        if is_master(args):
            log_master = Logger(args.logfile)
        else:
            log_master = logging

        logger = log_master.getLogger(f"main")

        if args.gpu is not None:
            logger.info("Using GPU: {}".format(args.gpu))

        if args.distributed:
            if args.dist_url == "env://" and args.rank == -1:
                args.rank = int(os.environ["RANK"])
            if args.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                args.rank = args.rank * ngpus_per_node + gpu
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)

        logger.info("Loading data...")
        train_loader, val_loader, aux_data = self.load_data(args, logger)
        logger.info("Setting up model/optim/etc...")
        model = self.setup_model(args, aux_data)
        model = self.setup_distributed(args, model)
        criterion = self.setup_criterion(args)
        optimizer = self.setup_optimizer(args, model)
        model, optimizer, start_epoch, checkpoint = self.resume_model(args, model, optimizer, logger)
        scheduler = self.setup_scheduler(args, optimizer)

        best_metric = None

        if args.evaluate:
            logger.info("Evaluate only...")
            _, outputs = self.eval_epoch_wrapper(val_loader, aux_data, model, criterion, start_epoch, args, log_master)
            self.save_output(outputs, f"output_eval.pkl", logger)
            return

        for epoch in range(start_epoch+1, args.epoch+1):
            if is_master(args):
                logger.info(f"Epoch {epoch:3} training...")
            self.pre_train_hook(train_loader, aux_data, model, criterion, optimizer, epoch, args, log_master)
            loss = self.train_epoch(train_loader, aux_data, model, criterion, optimizer, epoch, args, log_master)
            self.scheduler_step(args, scheduler, loss)
            self.pos_train_hook(train_loader, aux_data, model, criterion, optimizer, epoch, args, log_master)

            if args.eval_epoch > 0 and epoch % args.eval_epoch == 0:
                if is_master(args):
                    logger.info(f"Epoch {epoch:3} evaluating...")

                self.pre_eval_hook(val_loader, aux_data, model, criterion, epoch, args, log_master)
                metric, outputs = self.eval_epoch_wrapper(val_loader, aux_data, model, criterion, epoch, args, log_master)
                self.pos_eval_hook(val_loader, aux_data, model, criterion, epoch, args, log_master)
                
                if best_metric is None or metric > best_metric:
                    self.save_model(args, model, optimizer, epoch, metric, loss, is_best=True)
                    self.save_output(outputs, "output_best.pkl", logger)
                    best_metric = metric

                if self.args.save_all:
                    self.save_output(outputs, f"output_{epoch}.pkl", logger)

            if args.save_epoch > 0 and epoch % args.save_epoch == 0:
                self.save_model(args, model, optimizer, epoch, metric, loss)


    def pre_train_hook(self, *args, **kwargs):
        pass

    def pos_train_hook(self, *args, **kwargs):
        pass

    def pre_eval_hook(self, *args, **kwargs):
        pass

    def pos_eval_hook(self, *args, **kwargs):
        pass

    def load_data(self, args, logger=None):
        train_dataset = torchvision.datasets.MNIST(root='./data',
                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=True)

        train_sampler = get_sampler(args, train_dataset)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=False,
                                                   num_workers=0,
                                                   pin_memory=True,
                                                   sampler=train_sampler)

        return train_loader, train_loader, None

    def setup_model(self, args, aux_data):
        class ConvNet(nn.Module):
            def __init__(self, num_classes=10):
                super(ConvNet, self).__init__()
                self.layer1 = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2))
                self.layer2 = nn.Sequential(
                    nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2))
                self.fc = nn.Linear(7 * 7 * 32, num_classes)

            def forward(self, x):
                out = self.layer1(x)
                out = self.layer2(out)
                out = out.reshape(out.size(0), -1)
                out = self.fc(out)
                return out

        return ConvNet()

    def setup_criterion(self, args):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def setup_optimizer(self, args, model):
        params = filter(lambda p: p.requires_grad, model.parameters())
        if args.optimizer == 'adam':
            optimizer = optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay,
                                        betas=(args.beta1, args.beta2))
        elif args.optimizer == "rmsprop":
            optimizer = optim.RMSprop(params, lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(params, lr=args.learning_rate, momentum=args.alpha,
                                       weight_decay=args.weight_decay)
        else:
            raise ValueError(f"undefined optimizer: {args.optimizer}")
        return optimizer

    def setup_scheduler(self, args, optimizer):
        if args.scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_decay, patience=args.patience)
        elif args.scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.patience, gamma=args.lr_decay)
        elif args.scheduler == "multistep":
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.patience, gamma=args.lr_decay)
        elif args.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)
        else:
            scheduler = None
        return scheduler

    def scheduler_step(self, args, scheduler, loss):
        if isinstance(loss, dict):
            loss = loss.get("avg", 0)
        if args.scheduler == "plateau":
            scheduler.step(loss)
        elif scheduler is not None:
            scheduler.step()

    def resume_model(self, args, model, optimizer, logger):
        start_epoch = 0
        checkpoint = dict()
        if args.resume:
            if os.path.isfile(args.resume):
                logger.info("=> loading checkpoint '{}'".format(args.resume))
                if args.gpu is None:
                    checkpoint = torch.load(args.resume)
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(args.gpu)
                    checkpoint = torch.load(args.resume, map_location=loc)
                start_epoch = checkpoint['epoch']

                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                logger.info("=> loaded checkpoint '{}' (epoch {})"
                            .format(args.resume, checkpoint['epoch']))
            else:
                logger.info("=> no checkpoint found at '{}'".format(args.resume))
        return model, optimizer, start_epoch, checkpoint

    def save_model(self, args, model, optimizer, epoch, metric, loss, is_best=False):
        if is_best:
            ckpt_path = Path(args.paths["model_ckpts"]) / Path("model_best.pt")
        else:
            ckpt_path = Path(args.paths["model_ckpts"]) / Path("model_" + str(epoch) + ".pt")
        states = {
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "config": args,
            "metric": metric,
            "loss": loss,
            "epoch": epoch
        }
        torch.save(states, str(ckpt_path))

    def save_output(self, data, filename="output.pkl", logger=None):
        filepath = Path(self.args.paths["outputs"]) / Path(filename)
        with filepath.open("wb") as fout:
            pk.dump(data, fout)
        if logger and is_master(self.args):
            logger.info(f"Model output saved at: {filepath}")

    def train_epoch(self, train_loader, aux_data, model, criterion, optimizer, epoch, args, log_master):
        model.train()
        logger = log_master.get_logger(f"train")
        if is_master(args):
            logger.info("Start training...")

        start_time = time.time()
        running_loss = AverageMonitor("avg")
        max_batch = len(train_loader)
        if is_master(args):
            loader = tqdm(train_loader, leave=False, ncols=60, unit='b', total=max_batch)
        else:
            loader = train_loader

        overall_loss = AverageMeter("overall")
        for i, data in enumerate(loader):
            data = to_device(data, "cuda")
            aux_data = to_device(aux_data, "cuda")
            avg_loss, loss_dict= self.train_batch(data, aux_data, model, criterion, optimizer, i, args)
            loss_dict = to_item(loss_dict)
            avg_loss = to_item(avg_loss)

            if i % args.log_freq == 0 and is_master(args):
                mlflow.log_metrics(loss_dict)

            running_loss.update(loss_dict)
            overall_loss.update(avg_loss)

        elapsed = (time.time() - start_time) / 60
        if is_master(args):
            logger.info(f'Epoch: {epoch:3}, time elapsed: {elapsed:3.1f} mins, running loss: {running_loss}')
            mlflow.log_metrics(running_loss.get_average())
        return overall_loss.avg

    def train_batch(self, batch_data, aux_data, model, criterion, optimizer, batch_id, args):
        imgs, labels = batch_data
        logits = model(imgs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()
        loss_dict = {"loss", loss.item()}
        loss = loss.item()
        return loss, loss_dict

    def eval_epoch_wrapper(self, val_loader, aux_data, model, criterion, epoch, args, log_master):
        model.eval()
        logger = log_master.get_logger(f"eval")
        if is_master(args):
            logger.info("Start evaluating...")

        max_batch = len(val_loader)
        if is_master(args):
            loader = tqdm(val_loader, leave=False, ncols=70, unit='b', total=max_batch)
        else:
            loader = val_loader

        start_time = time.time()
        score, metrics, outputs = self.eval_epoch(loader, aux_data, model, criterion, epoch, args, log_master)
        elapsed = (time.time() - start_time) / 60

        if is_master(args):
            metrics2 = AverageMonitor("val")
            metrics2.update(metrics)
            metrics = metrics2.get_average()
            logger.info(f"Epoch {epoch:3}, time elapsed: {elapsed:3.1} mins, metrics: {metrics2}")
            mlflow.log_metrics(metrics, step=epoch)

        return score, outputs

    def eval_epoch(self, val_loader, aux_data, model, criterion, epoch, args, log_master):
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        with torch.no_grad():
            for batch in val_loader:
                batch = to_device(batch, model.device)
                images, labels = batch
                logits = model(images)
                loss = criterion(logits, labels)
                acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1, images.size(0))
                top5.update(acc5, images.size(0))

        loss = losses.avg
        top1 = top1.avg
        top5 = top5.avg
        metrics = {
            "loss": loss,
            "acc1": top1,
            "acc5": top5
        }
        return top1, metrics, {"output": None}

    def setup_distributed(self, args, model):
        ngpus_per_node = args.ngpus_per_node
        if args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        elif torch.cuda.device_count() == 1:
            model = model.cuda()
            print("Using single GPU since there is only one available")
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            print("Using DataParallel")
            model = torch.nn.DataParallel(model).cuda()

        return model
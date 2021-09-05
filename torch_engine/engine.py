

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import sys
from pathlib import Path
import datetime
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
import PIL
from pprint import pprint
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed as dist
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


from .experiment import Experiment
from .py_logger import Logger, log_args


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

        if is_master(self.args):
            mlflow.set_tracking_uri(self.mlflow_server)
            experiment_id = mlflow.set_experiment(str(self.mlflow_dir))
            with mlflow.start_run(experiment_id=experiment_id):
                mlflow.log_param("base_dir", str(self.base_dir))
                mlflow.log_param("comment", self.comment)
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
        model = setup_distributed(args, model)
        criterion = self.setup_criterion(args)
        optimizer = self.setup_optimizer(args, model)
        model, optimizer, start_epoch, checkpoint = self.resume_model(args, model, optimizer, logger)
        scheduler = self.setup_scheduler(args, optimizer)

        best_metric = -1

        if args.evaluate:
            logger.info("Evaluate only...")
            self.eval_epoch(val_loader, aux_data, model, criterion, start_epoch, args, log_master)
            return

        for epoch in range(start_epoch+1, args.epoch+1):
            if is_master(args):
                logger.info(f"Epoch {epoch:3} training...")
            self.pre_train_hook(train_loader, aux_data, model, criterion, optimizer, epoch, args, log_master)
            loss = self.train_epoch(train_loader, aux_data, model, criterion, optimizer, epoch, args, log_master)
            self.scheduler_step(args, scheduler, loss)
            self.pos_train_hook(train_loader, aux_data, model, criterion, optimizer, epoch, args, log_master)

            if is_master(args):
                logger.info(f"Epoch {epoch:3} evaluating...")

            self.pre_eval_hook(val_loader, aux_data, model, criterion, epoch, args, log_master)
            metric = self.eval_epoch_wrapper(val_loader, aux_data, model, criterion, epoch, args, log_master)
            self.pos_eval_hook(val_loader, aux_data, model, criterion, epoch, args, log_master)

            if args.save_epoch > 0 and epoch % args.save_epoch == 0:
                self.save_model(args, model, optimizer, epoch, metric, loss)

            if metric > best_metric:
                self.save_model(args, model, optimizer, epoch, metric, loss, is_best=True)
                best_metric = metric

    @staticmethod
    def pre_train_hook(*args, **kwargs):
        pass

    @staticmethod
    def pos_train_hook(*args, **kwargs):
        pass

    @staticmethod
    def pre_eval_hook(*args, **kwargs):
        pass

    @staticmethod
    def pos_eval_hook(*args, **kwargs):
        pass

    @staticmethod
    def load_data(args, logger=None):
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

    @staticmethod
    def setup_model(args, aux_data):
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

    @staticmethod
    def setup_criterion(args):
        criterion = nn.CrossEntropyLoss()
        return criterion

    @staticmethod
    def setup_optimizer(args, model):
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

    @staticmethod
    def setup_scheduler(args, optimizer):
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

    @staticmethod
    def scheduler_step(args, scheduler, loss):
        if isinstance(loss, dict):
            loss = loss.get("avg", 0)
        if args.scheduler == "plateau":
            scheduler.step(loss)
        elif scheduler is not None:
            scheduler.step()

    @staticmethod
    def resume_model(args, model, optimizer, logger):
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


    @staticmethod
    def save_model(args, model, optimizer, epoch, metric, loss, is_best=False):
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

    def train_epoch(self, train_loader, aux_data, model, criterion, optimizer, epoch, args, log_master):

        model.train()
        logger = log_master.get_logger(f"train")
        if is_master(args):
            logger.info("Start training...")

        start_time = time.time()
        running_loss = AverageMonitor("avg")
        max_batch = len(train_loader)
        if is_master(args):
            loader = tqdm(train_loader, leave=False, ncols=70, unit='b', total=max_batch)
        else:
            loader = train_loader

        overall_loss = AverageMeter("overall")
        for i, data in enumerate(loader):
            loss, avg_loss = self.train_batch(data, aux_data, model, criterion, optimizer, i, args)

            if i % args.log_freq == 0 and is_master(args):
                mlflow.log_metrics(loss)

            running_loss.update(loss)
            overall_loss.update(avg_loss)

        elapsed = (time.time() - start_time) / 60
        if is_master(args):
            logger.info(f'Epoch: {epoch:3}, time elapsed: {elapsed:3.1f} mins, running loss: {running_loss}')
            mlflow.log_metrics(running_loss.get_average())
        return overall_loss.avg

    @staticmethod
    def train_batch(batch_data, aux_data, model, criterion, optimizer, batch_id, args):
        return 0.0, 0.0

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

        return self.eval_epoch(loader, aux_data, model, criterion, epoch, args, log_master)

    @staticmethod
    def eval_epoch(val_loader, aux_data, model, criterion, epoch, args, log_master):
        return 0


class AverageMonitor(object):
    def __init__(self, name='', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.summation = defaultdict(lambda: 0)
        self.counts = defaultdict(lambda: 0)

    def update(self, new_data):
        for k, v in new_data.items():
            self.summation[k] += v
            self.counts[k] += 1

    def get_average(self):
        avg = {}
        for k, v in self.summation.items():
            if self.name == '':
                key = k
            else:
                key = self.name + '_' + k
            if self.counts[k] > 0:
                avg[key] = self.summation[k] / self.counts[k]
            else:
                avg[key] = -1
        return avg

    def __str__(self):
        avg = self.get_average()
        string = 'Avg: '
        for k, v in avg.items():
            string += f" {k}: {v:.4f};"
        return string[:-1]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, flush=True):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if flush:
            print('\t'.join(entries), flush=True)
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_sampler(args, dataset):
    if args.distributed:
        train_sampler = dist.DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank)
    else:
        train_sampler = None
    return train_sampler


def is_master(args):
    ngpus_per_node = args.ngpus_per_node
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        return True
    else:
        return False


def setup_distributed(args, model):
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
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print("Using DataParallel")
        model = torch.nn.DataParallel(model).cuda()

    return model
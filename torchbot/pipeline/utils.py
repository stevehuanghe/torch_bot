import torch.distributed as dist
import torch.utils.data.distributed as datadist

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_sampler(args, dataset):
    if args.distributed or dist.is_initialized():
        train_sampler = datadist.DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank)
    else:
        train_sampler = None
    return train_sampler


def is_master(args):
    if not dist.is_initialized():
        return True
    ngpus_per_node = args.ngpus_per_node
    if args.rank % ngpus_per_node == 0:
        return True
    else:
        return False

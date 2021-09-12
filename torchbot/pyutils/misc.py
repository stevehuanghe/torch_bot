import os
import warnings
from pathlib import Path
from collections import defaultdict


def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        # elif value is not None:
        else:
            dict_to[key] = dict_from[key]


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
        string = ''
        for k, v in avg.items():
            string += f" {k}: {v:.4f};"
        return string[:-1]


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


def mkdir_safe(dir):
    dir = Path(dir)
    if not dir.is_dir():
        try:
            dir.mkdir(parents=True)
        except OSError as e:
            import errno
            if e.errno == errno.EEXIST:
                warnings.warn("Failed creating path: {path}, probably a race"
                              " condition".format(path=str(dir)))

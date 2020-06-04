import os
from pathlib import Path
import warnings


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

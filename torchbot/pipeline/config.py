import argparse
import yaml
from pprint import pprint

class ArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        ################################ Protected #####################################
        parser.add_argument('-cf', '--config', metavar='FILE', default=None,
                            help='path to config file')
        parser.add_argument('--exp_name', metavar='STR', default=None,
                            help='mlflow experiment name')
        parser.add_argument('--run_name', metavar='STR', default=None,
                            help='mlflow experiment run name')
        parser.add_argument('-lg', '--log_dir', metavar='DIR', default='./outputs',
                            help='path to log directory')
        parser.add_argument('-od', '--output_dir', metavar='DIR', default='./outputs',
                            help='path to output directory')
        parser.add_argument('--seed', metavar='INT', default=None, type=int,
                            help='random seed')
        parser.add_argument('-bs', '--batch_size', metavar='INT', default=512, type=int,
                            help='batch size')
        parser.add_argument('-ep', '--epoch', metavar='INT', default=10, type=int,
                            help='max epoch to train')
        parser.add_argument('--log_freq', metavar='INT', default=5, type=int,
                            help='number of steps for logging training loss')
        parser.add_argument('--save_all', metavar='BOOL', default=False, type=bool,
                            help='save model output every time')
        parser.add_argument('--save_epoch', metavar='INT', default=0, type=int,
                            help='save model every number of epochs')
        parser.add_argument('--eval_epoch', metavar='INT', default=0, type=int,
                            help='eval model every number of epochs')
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--resume_epoch', default=True, type=bool, metavar='BOOL',
                            help='whether to load epoch (default: True)')
        parser.add_argument('--resume_optim', default=True, type=bool, metavar='BOOL',
                            help='whether to load optimizer (default: True)')
        parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                            help='evaluate model on validation set')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model')

        # for pytorch distributed
        parser.add_argument('--world_size', default=-1, type=int,
                            help='number of nodes for distributed training')
        parser.add_argument('--rank', default=-1, type=int,
                            help='node rank for distributed training')
        parser.add_argument('--dist_url', default='tcp://127.0.0.1:23457', type=str,
                            help='url used to set up distributed training')
        parser.add_argument('--dist_backend', default='nccl', type=str,
                            help='distributed backend')
        parser.add_argument('--gpu', default=None, type=int,
                            help='GPU id to use.')
        parser.add_argument('--cudnn_benchmark', action='store_true')
        parser.add_argument('--multiprocessing_distributed', action='store_true',
                            help='Use multi-processing distributed training to launch '
                                 'N processes per node, which has N GPUs. This is the '
                                 'fastest way to use PyTorch for either single node or '
                                 'multi node data parallel training')

        # dataloader
        parser.add_argument('--workers', metavar='INT', default=4, type=int,
                            help='number of wokers for dataloader')
        parser.add_argument('--pin_memory', metavar='BOOL', default=False, type=bool,
                            help='whether to pin memory')

        # optimization
        parser.add_argument('-lr', '--learning_rate', metavar='FLOAT', default=0.0001, type=float,
                            help='learning rate')
        parser.add_argument('-a', '--alpha', metavar='FLOAT', default=0.9, type=float,
                            help='hyper-param ')
        parser.add_argument('-b1', '--beta1', metavar='FLOAT', default=0.5, type=float,
                            help='hyper-param ')
        parser.add_argument('-b2', '--beta2', metavar='FLOAT', default=0.999, type=float,
                            help='hyper-param ')
        parser.add_argument('-dc', '--weight_decay', metavar='FLOAT', default=0.0, type=float,
                            help='weight normalization term')
        parser.add_argument('-opt', '--optimizer', metavar='STR', default='adam', type=str,
                            help='which optimizer to use')
        parser.add_argument('-gc', '--grad_clip', metavar='FLOAT', default=None, type=float,
                            help='clip grad norm')

        # lr scheduler
        parser.add_argument('--scheduler', default=None, type=str, metavar='MODE',
                            help='lr scheduler (default: none)')
        parser.add_argument('--patience', metavar='FLOAT', default=10.0, type=float,
                            help='patience/step_size for lr decay')
        parser.add_argument('--lr_decay', metavar='FLOAT', default=0.1, type=float,
                            help='lr decay factor')
        

        ############################## end of protected ################################

        ############################## custom args #####################################
        
       
        ############################## end of custom args ##############################

        self.parser = parser

        args = self.parser.parse_args()
        if args.config is not None:
            with open(args.config, 'r') as fin:
                options_yaml = yaml.load(fin, Loader=yaml.FullLoader)
            self.update_values(options_yaml, vars(args))
        self.args = args

    def update_values(self, dict_from, dict_to):
        for key, value in dict_from.items():
            if isinstance(value, dict):
                self.update_values(dict_from[key], dict_to[key])
            # elif value is not None:
            else:
                dict_to[key] = dict_from[key]

    def get_args(self):
        return self.args

    def save_args(self, filename):
        with open(filename, 'w') as out:
            yaml.dump(vars(self.args), out)

if __name__ == "__main__":
    config = ArgParser()
    config.log_args('args.yaml')

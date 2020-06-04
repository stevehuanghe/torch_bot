import argparse
import yaml
from pprint import pprint

class ArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        ################################ Protected #####################################
        parser.add_argument('-cf', '--config', metavar='FILE', default=None,
                            help='path to config file')
        parser.add_argument('-r', '--result', metavar='DIR', default='./result',
                            help='path to result directory')
        parser.add_argument('--seed', metavar='INT', default=None, type=int,
                            help='random seed')
        parser.add_argument('-bs', '--batch_size', metavar='INT', default=512, type=int,
                            help='batch size')
        parser.add_argument('-ep', '--epoch', metavar='INT', default=10, type=int,
                            help='max epoch to train')
        parser.add_argument('--log_freq', metavar='INT', default=5, type=int,
                            help='number of steps for logging training loss')
        parser.add_argument('--save_epoch', metavar='INT', default=0, type=int,
                            help='save model every number of epochs')
        parser.add_argument('--decay_epoch', metavar='INT', default=10, type=int,
                            help='decay learning rate every # epochs')
        parser.add_argument('--decay_mode', default=None, type=str, metavar='MODE',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
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
        parser.add_argument('--multiprocessing_distributed', action='store_true',
                            help='Use multi-processing distributed training to launch '
                                 'N processes per node, which has N GPUs. This is the '
                                 'fastest way to use PyTorch for either single node or '
                                 'multi node data parallel training')

        # training
        parser.add_argument('--workers', metavar='INT', default=2, type=int,
                            help='number of wokers for dataloader')

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
        parser.add_argument('-gc', '--grad_clip', metavar='FLOAT', default=10.0, type=float,
                            help='clip grad norm')
        ############################## end of protected ################################

        # data
        parser.add_argument('--dataset', default='mitstates', help='mitstates|zappos')
        parser.add_argument('--data_dir', default='data/mit-states', help='data root dir')
        parser.add_argument('--splitname', default='compositional-split-natural')
        parser.add_argument('--test_set', default='val', help='val|test')
        parser.add_argument('--pair_dropout', type=float, default=0.0,
                            help='Each epoch drop this fraction of train pairs')
        parser.add_argument('--pair_dropout_epoch', type=int, default=1,
                            help='Shuffle pair dropout every N epochs')
        parser.add_argument('--neg_ratio', type=float, default=0.25)
        parser.add_argument('--subset', action='store_true', default=False,
                            help='test on a 1000 image subset')
        parser.add_argument('--num_negs', type=int, default=1,
                            help='Number of negatives to sample per positive')
        parser.add_argument('--word_dim', type=int, default=100,
                            help='Dimension of word embeddings to load')
        parser.add_argument('--img_size', type=int, default=128,
                            help='Size of images')

        # model
        parser.add_argument('--model', metavar='STR', default='dcgan', type=str,
                            help='which model to use')

        parser.add_argument('--z_dim', metavar='INT', default=100, type=int,
                            help='noise dimension')
        parser.add_argument('--depth', metavar='INT', default=4, type=int,
                            help='number of layers')
        parser.add_argument('--dim_G', metavar='INT', default=64, type=int,
                            help='base dim for G')
        parser.add_argument('--dim_D', metavar='INT', default=64, type=int,
                            help='base dim for D')

        parser.add_argument('-bb', '--backbone', metavar='STR', default='resnet101', type=str,
                            help='which backbone to use')
        parser.add_argument('--acti', metavar='STR', default='relu', type=str,
                            help='which activation function to use')
        parser.add_argument('--n_neg', metavar='INT', default=5, type=int,
                            help='number of negative labels')
        parser.add_argument('--dropout', metavar='FLOAT', default=0.0, type=float,
                            help='dropout')
        parser.add_argument('-ls', '--loss_mode', metavar='STR', default='MSE', type=str,
                            help='which loss to use')
        parser.add_argument('--lambda_GP', metavar='FLOAT', default=0.0, type=float,
                            help='weight for WGAN-GP')
        parser.add_argument('--gan_cls', action='store_true', default=False,
                            help='test on a 1000 image subset')

        parser.add_argument('--n_iter', metavar='INT', default=5, type=int,
                            help='number of layers')
        parser.add_argument('--lr_G', metavar='FLOAT', default=0.001, type=float,
                            help='learning rate for generator')
        parser.add_argument('--lr_D', metavar='FLOAT', default=0.001, type=float,
                            help='learning rate for discriminator')
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

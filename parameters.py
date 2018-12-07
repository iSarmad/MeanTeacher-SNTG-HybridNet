
import argparse


def get_parameters():
    parser = argparse.ArgumentParser(description='Mean Teacher Trainer Pytorch')

    # TODO Attention , Please change the following parameters to reproduce Report results
    parser.add_argument('--sntg', default=True, help='Use SNTG loss?')  #
    parser.add_argument('--BN', default=True, help='Use Batch Normalization? ')
    parser.add_argument('--supervised_mode', default=False, type=bool, metavar='BOOL',
                        help='Training only with supervision')
    parser.add_argument('--lr', '--learning-rate', default=0.2, type=float,
                        metavar='LR', help='max learning rate')
    parser.add_argument('--lr_hybrid', default=0.2, type=float, help='max learning rate')


    # Additional Settings for Hybrid Network
    parser.add_argument('--initial_beta', default=0.9, help='beta for the adam optimizer')
    parser.add_argument('--model_hybrid',
                        default='hybridnet', help='Select Architecture for hybrid technique')

    #tensorboardX settings

    parser.add_argument('--saveX', default=True, help='Save for Tensorboard X ')
    parser.add_argument('--save_path', default='./ckpts', help='Path to Checkpoints of TensorboardX and models')
    parser.add_argument('--model',
                        default='convlarge', help='Basically using Convlarge for all experiments')
    parser.add_argument('-op', '--optim', default='Adam', help='Specify the Optimizer to use')
    parser.add_argument('--dataName',
                        default='cifar10', help='Name of Data used to train the models')

    parser.add_argument('--dataset', metavar='DATASET', default='cifar10')

    parser.add_argument('--datadir', type=str, default='data-local/images/cifar/cifar10/by-image',
                        help='data dir')
    parser.add_argument('--train-subdir', type=str, default='train',
                        help='the subdirectory inside the data directory that contains the training data')
    parser.add_argument('--eval-subdir', type=str, default='val',
                        help='the subdirectory inside the data directory that contains the evaluation data')
    parser.add_argument('--labels', default='data-local/labels/cifar10/4000_balanced_labels/00.txt', type=str, metavar='FILE',
                        help='list of image labels (default: based on directory structure)')

    parser.add_argument('--arch', '-a', metavar='ARCH', default='convlarge') # kind of redundant , remove it
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--labeled-batch-size', default=62, type=int,#62
                        metavar='N', help="labeled examples per minibatch (default: no constrain)")

    parser.add_argument('--initial-lr', default=0.0, type=float,
                        metavar='LR', help='initial learning rate when using linear rampup')
    parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                        help='length of learning rate rampup in the beginning')
    parser.add_argument('--lr-rampdown-epochs', default=210, type=int, metavar='EPOCHS',
                        help='length of learning rate cosine rampdown (>= length of training)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--beta', default=0.999, help='beta for the adam optimizer')

   # Optimizer Settings
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA',
                        help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--consistency', default=50.0/4, type=float, metavar='WEIGHT',# 100
                        help='use consistency loss with given weight (default: None)')
    parser.add_argument('--consistency-rampup', default=5, type=int, metavar='EPOCHS',
                        help='length of the consistency loss ramp-up')

    parser.add_argument('--checkpoint-epochs', default=100, type=int,
                        metavar='EPOCHS', help='checkpoint frequency in epochs, 0 to turn checkpointing off (default: 1)')
    parser.add_argument('--evaluation-epochs', default=1, type=int,
                        metavar='EPOCHS', help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', type=bool,
                        help='evaluate model on evaluation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--gpu_id', type=int, default=1, help='gpu ids: e.g. 0, 1. -1 is no GPU')

    return parser.parse_args()





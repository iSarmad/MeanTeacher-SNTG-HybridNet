import os
import shutil

import numpy as np
import torch

from Datasets.data import NO_LABEL
from misc.utils import *
from tensorboardX import SummaryWriter
import datetime
from parameters import get_parameters
import models

from misc import ramps
from Datasets import data
from models import losses

import torchvision.transforms as transforms


import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets

np.random.seed(5)
torch.manual_seed(5)

args =None


best_prec1 = 0
global_step = 0


def main(args):
    global global_step
    global best_prec1


    train_transform = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470,  0.2435,  0.2616))]))

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470,  0.2435,  0.2616))
    ])



    traindir = os.path.join(args.datadir, args.train_subdir)
    evaldir = os.path.join(args.datadir, args.eval_subdir)


    dataset = torchvision.datasets.ImageFolder(traindir, train_transform)

    if args.labels:
        with open(args.labels) as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())
        labeled_idxs, unlabeled_idxs = data.relabel_dataset(dataset, labels)


    if args.labeled_batch_size:
        batch_sampler = data.TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(evaldir, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)


#   Intializing the model
    model = models.__dict__[args.model](args, data=None).cuda()
    ema_model = models.__dict__[args.model](args,nograd = True, data=None).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    if args.evaluate:
        print('Evaluating the primary model')
        acc1 = validate(eval_loader, model)
        print('Accuracy of the Student network on the 10000 test images: %d %%' % (
                acc1))
        print('Evaluating the Teacher model')
        acc2 = validate(eval_loader, ema_model)
        print('Accuracy of the Teacher network on the 10000 test images: %d %%' % (
                acc2))
        return

    if args.saveX == True:
        save_path = '{},{},{}epochs,b{},lr{}'.format(
            args.model,
            args.optim,
            args.epochs,
            args.batch_size,
            args.lr)
        time_stamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        save_path = os.path.join(time_stamp, save_path)
        save_path = os.path.join(args.dataName, save_path)
        save_path = os.path.join(args.save_path, save_path)
        print('==> Will save Everything to {}', save_path)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

    test_writer = SummaryWriter(os.path.join(save_path, 'test'))


    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, ema_model, optimizer, epoch)


        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            prec1 = validate(eval_loader, model)

            ema_prec1 = validate(eval_loader, ema_model)

            print('Accuracy of the Student network on the 10000 test images: %d %%' % (
                prec1))
            print('Accuracy of the Teacher network on the 10000 test images: %d %%' % (
                ema_prec1))
            test_writer.add_scalar('Accuracy Student', prec1, epoch)
            test_writer.add_scalar('Accuracy Teacher', ema_prec1, epoch)

            is_best = ema_prec1 > best_prec1
            best_prec1 = max(ema_prec1, best_prec1)
        else:
            is_best = False

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.model,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, save_path, epoch + 1)




def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(train_loader, model, ema_model, optimizer, epoch):
    global global_step
    lossess = AverageMeter()
    running_loss = 0.0

    class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL).cuda()

    consistency_criterion = losses.softmax_mse_loss

    model.train()
    ema_model.train()

    for i, ((input, ema_input), target) in enumerate(train_loader):

        if (input.size(0) != args.batch_size):
            continue

        input_var = torch.autograd.Variable(input).cuda()


        target_var = torch.autograd.Variable(target.cuda(async=True))

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0

        if args.sntg == True:
            model_out,model_h = model(input_var)
        else:
            model_out = model(input_var)




        class_loss = class_criterion(model_out, target_var) / minibatch_size

        if not args.supervised_mode:
            with torch.no_grad():
                ema_input_var = torch.autograd.Variable(ema_input)
                ema_input_var = ema_input_var.cuda()

            if args.sntg == True:
                ema_model_out,ema_h = ema_model(ema_input_var)

            else:
                ema_model_out = ema_model(ema_input_var)

            ema_logit = ema_model_out

            ema_logit = Variable(ema_logit.detach().data, requires_grad=False)

            if args.consistency:
                if args.sntg: # implementation of SNTG loss
                    indx = (target == -1)
                    new_target = target.clone().cuda()
                    _, new_pred = ema_model_out.max(dim=1)
                    new_pred = new_pred.cuda()
                    new_target[indx] = new_pred[indx]
                    new_target_h1 = new_target[:minibatch_size // 2]
                    new_target_h2 = new_target[minibatch_size // 2:]

                    model_h1 = model_h[:minibatch_size // 2]
                    model_h2 = model_h[minibatch_size // 2:]
                    maskn = (new_target_h1 == new_target_h2)
                    maskn = maskn.float()
                    temp = torch.sum((model_h1 - model_h2) ** 2, 1)
                    pos = temp *maskn
                    neg = 1.0 - (1-maskn)*temp**(1/2)
                    neg = torch.clamp(neg, min=0) ** 2
                    sntg_loss = torch.sum(pos+ neg) / 128
                    consistency_weight = get_current_consistency_weight(epoch)
                    consistency_loss = consistency_weight * consistency_criterion(model_out, ema_logit) / minibatch_size
                    sntg_loss = (0.001/2) * sntg_loss * consistency_weight# best 0.001/2
                else:
                    consistency_weight = get_current_consistency_weight(epoch)
                    consistency_loss = consistency_weight * consistency_criterion(model_out, ema_logit) / minibatch_size
            else:
                consistency_loss = 0

            if args.sntg:


                loss = class_loss + consistency_loss + sntg_loss#.squeeze()
            else:
                loss = class_loss + consistency_loss
        else:
            loss = class_loss
        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        update_ema_variables(model, ema_model, args.ema_decay, global_step)

        # print statistics
        running_loss += loss.item()

        if i % 20 == 19:    # print every 20 mini-batches
            print('[Epoch: %d, Iteration: %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

        lossess.update(loss.item(), input.size(0))

    return lossess,running_loss


def validate(eval_loader, model):

    model.eval()
    total =0
    correct = 0
    for i, (input, target) in enumerate(eval_loader):

        with torch.no_grad():
            input_var = input.cuda()
            target_var = target.cuda(async=True)

            labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
            assert labeled_minibatch_size > 0
            if args.sntg == True:
                output1,_ = model(input_var)

            else:
                 # compute output
                output1 = model(input_var)

            _, predicted = torch.max(output1.data, 1)
            total += target_var.size(0)
            correct += (predicted == target_var).sum().item()

    return 100 * correct / total


def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        print('Best Model Saved: ');print(best_path)



def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)




if __name__ == '__main__':
    args = get_parameters()

    args.device = torch.device(
        "cuda:%d" % (args.gpu_id) if torch.cuda.is_available() else "cpu")

    main(args)

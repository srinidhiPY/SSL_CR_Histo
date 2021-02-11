"""
Task-Specific consistency training on downstream task (Camelyon16)
"""
import argparse
import os
import time
import random
import numpy as np
from PIL import Image
import cv2
import copy
from tqdm import tqdm
import torch.backends.cudnn as cudnn

import torch
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
from util import AverageMeter, plot_confusion_matrix
from collections import OrderedDict
from torchvision import transforms, datasets
import torch.nn.functional as F

from dataset import DatasetCamelyon16_Supervised_train, DatasetCamelyon16_SSLtrain, DatasetCamelyon16_eval, TransformFix
import models.net as net
from albumentations import Compose
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data.sampler import SubsetRandomSampler

##########
def train(args, model_teacher, model_student, classifier_teacher, classifier_student, tumor_labeled_train_loader, normal_labeled_train_loader, tumor_unlabeled_train_loader, normal_unlabeled_train_loader, optimizer, epoch):

    model_teacher.eval()
    classifier_teacher.eval()

    model_student.train()
    classifier_student.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    acc = AverageMeter()

    total_feats = []
    total_targets = []

    end = time.time()

    train_loader = zip(tumor_labeled_train_loader, normal_labeled_train_loader, tumor_unlabeled_train_loader,
                       normal_unlabeled_train_loader)

    for batch_idx, (tumor_data_x, normal_data_x, tumor_data_u, normal_data_u) in enumerate(
            tqdm(train_loader, disable=False)):

        # Get inputs and target
        tumor_inputs_x, tumor_targets_x = tumor_data_x  # tumor Labeled
        tumor_inputs_x = tumor_inputs_x.reshape(-1, 3, args.image_size, args.image_size)
        tumor_targets_x = tumor_targets_x.reshape(-1, )

        normal_inputs_x, normal_targets_x = normal_data_x  # normal Labeled
        normal_inputs_x = normal_inputs_x.reshape(-1, 3, args.image_size, args.image_size)
        normal_targets_x = normal_targets_x.reshape(-1, )

        tumor_inputs_u_w, tumor_inputs_u_s = tumor_data_u  # tumor Unlabeled
        normal_inputs_u_w, normal_inputs_u_s = normal_data_u  # normal Unlabeled

        tumor_inputs_x, normal_inputs_x, tumor_inputs_u_w, normal_inputs_u_w, tumor_inputs_u_s, normal_inputs_u_s, tumor_targets_x, normal_targets_x = tumor_inputs_x.float(), normal_inputs_x.float(), tumor_inputs_u_w.float(), normal_inputs_u_w.float(), \
                                                                                                                                                       tumor_inputs_u_s.float(), normal_inputs_u_s.float(), tumor_targets_x.long(), normal_targets_x.long()

        # Move the variables to Cuda
        tumor_inputs_x, normal_inputs_x, tumor_inputs_u_w, normal_inputs_u_w, tumor_inputs_u_s, normal_inputs_u_s, tumor_targets_x, normal_targets_x = tumor_inputs_x.cuda(), normal_inputs_x.cuda(), tumor_inputs_u_w.cuda(), normal_inputs_u_w.cuda(), \
                                                                                                                                                       tumor_inputs_u_s.cuda(), normal_inputs_u_s.cuda(), tumor_targets_x.cuda(), normal_targets_x.cuda()

        # Concatenate tumor and normal data and shuffle it
        shuffle_idx_x = torch.randperm(2 * len(tumor_inputs_x))
        shuffle_idx_u_w = torch.randperm(2 * len(tumor_inputs_u_w))
        shuffle_idx_u_s = torch.randperm(2 * len(tumor_inputs_u_s))

        inputs_x = torch.cat([tumor_inputs_x, normal_inputs_x])
        inputs_u_w = torch.cat([tumor_inputs_u_w, normal_inputs_u_w])
        inputs_u_s = torch.cat([tumor_inputs_u_s, normal_inputs_u_s])
        targets_x = torch.cat([tumor_targets_x, normal_targets_x])

        # shuffle
        inputs_x = inputs_x[shuffle_idx_x, :, :, :]
        inputs_u_w = inputs_u_w[shuffle_idx_u_w, :, :, :]
        inputs_u_s = inputs_u_s[shuffle_idx_u_s, :, :, :]
        targets_x = targets_x[shuffle_idx_x]

        # Compute pseudolabels for weak_unlabeled images using the teacher model
        with torch.no_grad():
            feat_u_w = model_teacher(inputs_u_w)  # weak unlabeled data
            logits_u_w = classifier_teacher(feat_u_w)

        # Compute output for labeled and strong_unlabeled images using the student model
        inputs = torch.cat((inputs_x, inputs_u_s))
        feats = model_student(inputs)
        logits = classifier_student(feats)

        batch_size = inputs_x.shape[0]
        logits_x = logits[:batch_size]  # labeled data
        logits_u_s = logits[batch_size:]  # unlabeled data
        del logits

        # Compute loss
        Supervised_loss = F.cross_entropy(logits_x, targets_x, reduction='mean')

        pseudo_label = torch.softmax(logits_u_w.detach_(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        Consistency_loss = F.cross_entropy(logits_u_s, targets_u, reduction='mean')

        final_loss = Supervised_loss + args.lambda_u * Consistency_loss

        # compute gradient and do SGD step #############
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        # compute loss and accuracy ####################
        losses_x.update(Supervised_loss.item(), batch_size)
        losses_u.update(Consistency_loss.item(), batch_size)
        losses.update(final_loss.item(), batch_size)
        pred = torch.argmax(logits_x, dim=1)
        acc.update(torch.sum(targets_x == pred).item() / batch_size, batch_size)

        # Save features
        total_feats.append(feats[:batch_size])
        total_targets.append(targets_x)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print statistics and write summary every N batch
        if (batch_idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'acc {acc.val:.3f} ({acc.avg:.3f})\t'
                  'final_loss {final_loss.val:.3f} ({final_loss.avg:.3f})\t'
                  'Supervised_loss {Supervised_loss.val:.3f} ({Supervised_loss.avg:.3f})\t'
                  'Consistency_loss {Consistency_loss.val:.3f} ({Consistency_loss.avg:.3f})'.format(epoch, batch_idx + 1, (len(tumor_labeled_train_loader) + len(normal_labeled_train_loader)),
                                                                                                    batch_time=batch_time,
                                                                                                    data_time=data_time,
                                                                                                    acc=acc,
                                                                                                    final_loss=losses,
                                                                                                    Supervised_loss=losses_x,
                                                                                                    Consistency_loss=losses_u))

        final_feats = torch.cat(total_feats).detach()
        final_targets = torch.cat(total_targets).detach()

    return losses.avg, losses_x.avg, losses_u.avg, acc.avg, final_feats, final_targets


def validate(args, model_student, classifier_student, val_tumor_loader, val_normal_loader, epoch):

    # switch to evaluate mode
    model_student.eval()
    classifier_student.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    with torch.no_grad():

        end = time.time()

        val_loader = zip(val_tumor_loader, val_normal_loader)

        for batch_idx, (data_tumor, data_normal) in enumerate(tqdm(val_loader, disable=False)):

            # Get inputs and target
            tumor_inputs_x, tumor_targets_x = data_tumor
            normal_inputs_x, normal_targets_x = data_normal

            # Concatenate tumor and normal data and shuffle it
            shuffle_idx_x = torch.randperm(2 * len(tumor_inputs_x))
            input = torch.cat([tumor_inputs_x, normal_inputs_x])
            target = torch.cat([tumor_targets_x, normal_targets_x])

            # shuffle
            input = input[shuffle_idx_x, :, :, :]
            target = target[shuffle_idx_x]

            # Get inputs and target
            input, target = input.float(), target.long()

            # Move the variables to Cuda
            input, target = input.cuda(), target.cuda()

            # compute output ###############################
            feats = model_student(input)
            output = classifier_student(feats)

            loss = F.cross_entropy(output, target, reduction='mean')

            # compute loss and accuracy ####################
            batch_size = target.size(0)
            losses.update(loss.item(), batch_size)

            pred = torch.argmax(output, dim=1)
            acc.update(torch.sum(target == pred).item() / batch_size, batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print statistics and write summary every N batch
            if (batch_idx + 1) % args.print_freq == 0:
                print('Val: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'acc {acc.val:.3f} ({acc.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})'.format(epoch, batch_idx + 1, 2 * len(val_tumor_loader),
                                                                    batch_time=batch_time,
                                                                    data_time=data_time, acc=acc, loss=losses))

        return losses.avg, acc.avg


def parse_args():

    parser = argparse.ArgumentParser('Argument for Camelyon16 - Consistency training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--gpu', default='0, 1', help='GPU id to use.')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use.')
    parser.add_argument('--seed', type=int, default=42, help='seed for initializing training.')

    # model definition
    parser.add_argument('--model', type=str, default='resnet18', help='choice of network architecture.')
    parser.add_argument('--mode', type=str, default='fine-tuning', help='fine-tuning')
    parser.add_argument('--modules_teacher', type=int, default=64,
                        help='which modules to freeze for the fine-tuned teacher model. (full-finetune(0), fine-tune only FC layer (60). Full_network(64) - Resnet18')
    parser.add_argument('--modules_student', type=int, default=60,
                        help='which modules to freeze for fine-tuning the student model. (full-finetune(0), fine-tune only FC layer (60) - Resnet18')
    parser.add_argument('--num_classes', type=int, default=2, help='# of classes.')
    parser.add_argument('--num_epoch', type=int, default=90, help='epochs to train for - 150.')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size - 8/16.')
    parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size - 7')
    parser.add_argument('--NAug', default=7, type=int, help='No of Augmentations for strong unlabeled data')

    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate. - 5e-4(SGD)')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay/weights regularizer for sgd. - 1e-4')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum for sgd, beta1 for adam.')
    parser.add_argument('--beta2', default=0.999, type=float, help=' beta2 for adam.')
    parser.add_argument('--lambda_u', default=1, type=float, help='coefficient of unlabeled loss')

    parser.add_argument('--model_path_finetune', type=str,
                        default='/home/cspy87/projects/rrg-amartel/cspy87/Camelyon16/Fine-tune/SSL/0.1/',
                        help='path to load SSL fine-tuned model to intialize "Teacher and student network" for consistency training')
    parser.add_argument('--model_save_pth', type=str,
                        default='/home/srinidhi/Research/Code/SSL_Resolution/Save_Results/Camelyon16/Fine-tune/SSL_CR/0.1/',
                        help='path to save fine-tuned model')
    parser.add_argument('--save_loss', type=str, default='/home/srinidhi/Research/Code/SSL_Resolution/Save_Results/Camelyon16/Fine-tune/SSL_CR/0.1/',
                        help='path to save loss and other performance metrics')
    parser.add_argument('--resume', type=str, default='/home/srinidhi/Research/Code/SSL_Resolution/Save_Results/Camelyon16/Fine-tune/SSL_CR/0.1/',
                        metavar='PATH', help='path to latest checkpoint - model.pth (default: none)')

    # Data paths
    parser.add_argument('--train_tumor_image_pth', default='/home/srinidhi/Research/Data/CAMELYON16/Fine_tune/PATCHES_TUMOR_TRAIN/')
    parser.add_argument('--train_normal_image_pth', default='/home/srinidhi/Research/Data/CAMELYON16/Fine_tune/PATCHES_NORMAL_TRAIN/')
    parser.add_argument('--json_train_pth', default='/home/srinidhi/Research/Data/CAMELYON16/Fine_tune/jsons/train/')
    parser.add_argument('--labeled_train', default=0.1, type=float, help='portion of the train data with labels - 1(full), 0.1/0.25/0.5')

    parser.add_argument('--val_tumor_image_pth', default='/home/srinidhi/Research/Data/CAMELYON16/Fine_tune/PATCHES_TUMOR_VALID/')
    parser.add_argument('--val_normal_image_pth', default='/home/srinidhi/Research/Data/CAMELYON16/Fine_tune/PATCHES_NORMAL_VALID/')
    parser.add_argument('--json_val_pth', default='/home/srinidhi/Research/Data/CAMELYON16/Fine_tune/jsons/valid/')

    # Tiling parameters
    parser.add_argument('--image_size', default=256, type=int, help='patch size width 256')

    args = parser.parse_args()

    return args


def main():

    # parse the args
    args = parse_args()

    # Set the data loaders (train, val, test)

    ### Camelyon16 #######

    if args.mode == 'fine-tuning':

        # Train set
        train_tumor_labeled_dataset = DatasetCamelyon16_Supervised_train(args.train_tumor_image_pth, args.json_train_pth)
        train_normal_labeled_dataset = DatasetCamelyon16_Supervised_train(args.train_normal_image_pth, args.json_train_pth)

        train_tumor_unlabeled_dataset = DatasetCamelyon16_SSLtrain(args.train_tumor_image_pth, args.json_train_pth, transform=TransformFix(args.image_size, args.NAug))
        train_normal_unlabeled_dataset = DatasetCamelyon16_SSLtrain(args.train_normal_image_pth, args.json_train_pth, transform=TransformFix(args.image_size, args.NAug))

        # Validation set
        val_tumor_dataset = DatasetCamelyon16_eval(args.val_tumor_image_pth, args.json_val_pth)
        val_normal_dataset = DatasetCamelyon16_eval(args.val_normal_image_pth, args.json_val_pth)

        # train and validation split
        train_tumor_idx = list(range(train_tumor_labeled_dataset.num_image))
        train_normal_idx = list(range(train_normal_labeled_dataset.num_image))

        valid_tumor_idx = list(range(val_tumor_dataset.num_image))
        valid_normal_idx = list(range(val_normal_dataset.num_image))

        #### Semi-Supervised Split (10, 25, 50, 100)
        tumor_labeled_train_idx = np.random.choice(train_tumor_idx, int(args.labeled_train * len(train_tumor_idx)))
        normal_labeled_train_idx = np.random.choice(train_normal_idx, int(args.labeled_train * len(train_normal_idx)))

        tumor_unlabeled_train_sampler = SubsetRandomSampler(train_tumor_idx)
        normal_unlabeled_train_sampler = SubsetRandomSampler(train_normal_idx)

        tumor_labeled_train_sampler = SubsetRandomSampler(tumor_labeled_train_idx)
        normal_labeled_train_sampler = SubsetRandomSampler(normal_labeled_train_idx)

        val_tumor_sampler = SubsetRandomSampler(valid_tumor_idx)
        val_normal_sampler = SubsetRandomSampler(valid_normal_idx)

        # Data loaders
        tumor_labeled_train_loader = torch.utils.data.DataLoader(train_tumor_labeled_dataset,
                                                                 batch_size=args.batch_size,
                                                                 sampler=tumor_labeled_train_sampler,
                                                                 shuffle=True if tumor_labeled_train_sampler is None else False,
                                                                 num_workers=args.num_workers, pin_memory=True,
                                                                 drop_last=True)
        normal_labeled_train_loader = torch.utils.data.DataLoader(train_normal_labeled_dataset,
                                                                  batch_size=args.batch_size,
                                                                  sampler=normal_labeled_train_sampler,
                                                                  shuffle=True if normal_labeled_train_sampler is None else False,
                                                                  num_workers=args.num_workers, pin_memory=True,
                                                                  drop_last=True)

        tumor_unlabeled_train_loader = torch.utils.data.DataLoader(train_tumor_unlabeled_dataset,
                                                                   batch_size=args.batch_size * args.mu,
                                                                   sampler=tumor_unlabeled_train_sampler,
                                                                   shuffle=True if tumor_unlabeled_train_sampler is None else False,
                                                                   num_workers=args.num_workers, pin_memory=True,
                                                                   drop_last=True)

        normal_unlabeled_train_loader = torch.utils.data.DataLoader(train_normal_unlabeled_dataset,
                                                                    batch_size=args.batch_size * args.mu,
                                                                    sampler=normal_unlabeled_train_sampler,
                                                                    shuffle=True if normal_unlabeled_train_sampler is None else False,
                                                                    num_workers=args.num_workers, pin_memory=True,
                                                                    drop_last=True)

        val_tumor_loader = torch.utils.data.DataLoader(val_tumor_dataset, batch_size=args.batch_size,
                                                       sampler=val_tumor_sampler, shuffle=False,
                                                       num_workers=args.num_workers, pin_memory=True, drop_last=False)

        val_normal_loader = torch.utils.data.DataLoader(val_normal_dataset, batch_size=args.batch_size,
                                                        sampler=val_normal_sampler, shuffle=False,
                                                        num_workers=args.num_workers, pin_memory=True, drop_last=False)

        # num of samples
        num_label_data = len(tumor_labeled_train_sampler)
        print('number of labeled tumor training samples: {}'.format(num_label_data))

        num_label_data = len(normal_labeled_train_sampler)
        print('number of labeled normal training samples: {}'.format(num_label_data))

        num_unlabel_data = len(tumor_unlabeled_train_sampler)
        print('number of unlabeled tumor training samples: {}'.format(num_unlabel_data))

        num_unlabel_data = len(normal_unlabeled_train_sampler)
        print('number of unlabeled normal training samples: {}'.format(num_unlabel_data))

        num_val_data = len(val_tumor_sampler)
        print('number of validation tumor samples: {}'.format(num_val_data))

        num_val_data = len(val_normal_sampler)
        print('number of validation normal samples: {}'.format(num_val_data))

    else:
        raise NotImplementedError('invalid mode {}'.format(args.mode))

    ########################################

  # set the model
    if args.model == 'resnet18':

        model_teacher = net.TripletNet_Finetune(args.model)
        model_student = net.TripletNet_Finetune(args.model)

        classifier_teacher = net.FinetuneResNet(args.num_classes)
        classifier_student = net.FinetuneResNet(args.num_classes)

        if args.mode == 'fine-tuning':

            ###### Intialize both teacher and student network with fine-tuned SSL model ###############

            ## Load teacher model ############

            # original model saved file with DataParallel (Multi-GPU)
            state_dict = torch.load(args.model_path_finetune)

            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()

            for k, v in state_dict['model'].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            # Load fine-tuned model
            print('==> loading pre-trained model')
            model_teacher.load_state_dict(new_state_dict)

            # look at the contents of the model and its parameters
            idx = 0
            for layer_name, param in model_teacher.named_parameters():
                print(layer_name, '-->', idx)
                idx += 1

            # Freeze the teacher model
            for name, param in enumerate(model_teacher.named_parameters()):
                if name < args.modules_teacher:  # No of layers(modules) to be freezed
                    print("module", name, "was frozen")
                    param = param[1]
                    param.requires_grad = False
                else:
                    print("module", name, "was not frozen")
                    param = param[1]
                    param.requires_grad = True

            # Load fine-tuned classifier
            print('==> loading pre-trained classifier')

            # create new OrderedDict that does not contain `module.`
            new_state_dict_CLS = OrderedDict()

            for k, v in state_dict['classifier'].items():
                name = k[7:]  # remove `module.`
                new_state_dict_CLS[name] = v

            classifier_teacher.load_state_dict(new_state_dict_CLS)

            ###### Load student model ############################

            # original model saved file with DataParallel (Multi-GPU)
            state_dict = torch.load(args.model_path_finetune)

            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()

            for k, v in state_dict['model'].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            # Load fine-tuned model
            print('==> loading pre-trained model')
            model_student.load_state_dict(new_state_dict)

            # look at the contents of the model and its parameters
            idx = 0
            for layer_name, param in model_student.named_parameters():
                print(layer_name, '-->', idx)
                idx += 1

            # Freeze the teacher model
            for name, param in enumerate(model_student.named_parameters()):
                if name < args.modules_student:  # No of layers(modules) to be freezed
                    print("module", name, "was frozen")
                    param = param[1]
                    param.requires_grad = False
                else:
                    print("module", name, "was not frozen")
                    param = param[1]
                    param.requires_grad = True

            # Load fine-tuned classifier
            print('==> loading pre-trained classifier')

            # create new OrderedDict that does not contain `module.`
            new_state_dict_CLS = OrderedDict()

            for k, v in state_dict['classifier'].items():
                name = k[7:]  # remove `module.`
                new_state_dict_CLS[name] = v

            classifier_student.load_state_dict(new_state_dict_CLS)

            # Multi-GPU
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                model_teacher = torch.nn.DataParallel(model_teacher)
                model_student = torch.nn.DataParallel(model_student)
                classifier_teacher = torch.nn.DataParallel(classifier_teacher)
                classifier_student = torch.nn.DataParallel(classifier_student)

        else:
            raise NotImplementedError('invalid training {}'.format(args.mode))

    else:
        raise NotImplementedError('model not supported {}'.format(args.model))

    # Load model to CUDA
    if torch.cuda.is_available():
        model_teacher = model_teacher.cuda()
        model_student = model_student.cuda()
        classifier_teacher = classifier_teacher.cuda()
        classifier_student = classifier_student.cuda()
        cudnn.benchmark = True

    # Optimiser & scheduler
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, list(model_student.parameters()) + list(classifier_student.parameters())), lr=args.lr, momentum=args.beta1, weight_decay=args.weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

    # Training Model
    start_epoch = 1
    best_val_acc = -1

    'check resume from a checkpoint'
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_student.load_state_dict(checkpoint['model_student'])
            model_teacher.load_state_dict(checkpoint['model_teacher'])
            classifier_teacher.load_state_dict(checkpoint['classifier_teacher'])
            classifier_student.load_state_dict(checkpoint['classifier_student'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['val_acc']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Start log (writing into XL sheet)
    with open(os.path.join(args.save_loss, 'fine_tuned_results.csv'), 'w') as f:
        f.write('epoch, train_loss, train_losses_x, train_losses_u, train_acc, val_loss, val_acc\n')

    # Routine
    for epoch in range(start_epoch, args.num_epoch + 1):

        if args.mode == 'fine-tuning':

            print("==> fine-tuning the SSL model...")

            time_start = time.time()

            train_losses, train_losses_x, train_losses_u, train_acc, final_feats, final_targets = train(args, model_teacher, model_student, classifier_teacher, classifier_student, tumor_labeled_train_loader, normal_labeled_train_loader, tumor_unlabeled_train_loader,
                                                                                                        normal_unlabeled_train_loader, optimizer, epoch)
            print('Epoch time: {:.2f} s.'.format(time.time() - time_start))

            print("==> validating the fine-tuned model...")
            val_losses, val_acc, = validate(args, model_student, classifier_student, val_tumor_loader, val_normal_loader, epoch)

            # Log results
            with open(os.path.join(args.save_loss, 'fine_tuned_results.csv'), 'a') as f:
                f.write('%03d, %0.6f, %0.6f, %0.6f, %0.6f, %0.6f, %0.6f,\n' % (
                    (epoch + 1), train_losses, train_losses_x, train_losses_u, train_acc, val_losses, val_acc,))

            'adjust learning rate --- Note that step should be called after validate()'
            scheduler.step()

            # Iterative training: Use the student as a teacher after every epoch
            model_teacher = copy.deepcopy(model_student)
            classifier_teacher = copy.deepcopy(classifier_student)

            # Save model every 10 epochs
            if epoch % args.save_freq == 0:
                print('==> Saving...')
                state = {
                    'args': args,
                    'model_student': model_student.state_dict(),
                    'model_teacher': model_teacher.state_dict(),
                    'classifier_teacher': classifier_teacher.state_dict(),
                    'classifier_student': classifier_student.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_losses,
                    'train_losses_x': train_losses_x,
                    'train_losses_u': train_losses_u,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'val_loss': val_losses,
                }
                torch.save(state, '{}/fine_tuned_model_{}.pt'.format(args.model_save_pth, epoch))

                # help release GPU memory
                del state
                torch.cuda.empty_cache()

            # Save model for the best val
            if val_acc > best_val_acc:
                print('==> Saving...')
                state = {
                    'args': args,
                    'model_student': model_student.state_dict(),
                    'model_teacher': model_teacher.state_dict(),
                    'classifier_teacher': classifier_teacher.state_dict(),
                    'classifier_student': classifier_student.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_losses,
                    'train_losses_x': train_losses_x,
                    'train_losses_u': train_losses_u,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'val_loss': val_losses,
                }
                torch.save(state, '{}/best_fine_tuned_model_{}.pt'.format(args.model_save_pth, epoch))
                best_val_acc = val_acc

                # help release GPU memory
                del state
                torch.cuda.empty_cache()

        else:
            raise NotImplementedError('mode not supported {}'.format(args.mode))


if __name__ == "__main__":

    args = parse_args()
    print(vars(args))

    # Force the pytorch to create context on the specific device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.gpu:
            torch.cuda.manual_seed_all(args.seed)

    # Main function
    main()

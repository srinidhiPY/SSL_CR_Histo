"""
Task-Specific consistency training on downstream task (Kather)
"""
import argparse
import os
import time
import random
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import torch.backends.cudnn as cudnn

import torch
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from util import AverageMeter, plot_confusion_matrix
from collections import OrderedDict
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler

from dataset import DatasetKather_SSLtrain, DatasetKather_eval, DatasetKather_Supervised_train, TransformFix
import models.net as net
from albumentations import Compose
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from albumentations import Compose, Rotate, CenterCrop, HorizontalFlip, RandomScale, Flip, Resize, ShiftScaleRotate, \
    RandomCrop, IAAAdditiveGaussianNoise, ElasticTransform, HueSaturationValue, LongestMaxSize, RandomBrightnessContrast


###########
def train(args, model_teacher, model_student, classifier_teacher, classifier_student, train_labeled_loader, train_unlabeled_loader, optimizer, epoch):

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

    end = time.time()

    train_loader = zip(train_labeled_loader, train_unlabeled_loader)

    for batch_idx, (data_x, data_u) in enumerate(tqdm(train_loader, disable=False)):

        # Get inputs and target
        inputs_x, targets_x = data_x
        inputs_u_w, inputs_u_s = data_u

        inputs_x, inputs_u_w, inputs_u_s, targets_x = inputs_x.float(), inputs_u_w.float(), inputs_u_s.float(), targets_x.long()

        # Move the variables to Cuda
        inputs_x, inputs_u_w, inputs_u_s, targets_x = inputs_x.cuda(), inputs_u_w.cuda(), inputs_u_s.cuda(), targets_x.cuda()

        # Compute output
        inputs_x = inputs_x.reshape(-1, 3, 256, 256)  #Reshape
        targets_x = targets_x.reshape(-1, )

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
                  'Consistency_loss {Consistency_loss.val:.3f} ({Consistency_loss.avg:.3f})'.format(epoch, batch_idx + 1, len(train_labeled_loader),
                                                                                                    batch_time=batch_time,
                                                                                                    data_time=data_time,
                                                                                                    acc=acc,
                                                                                                    final_loss=losses,
                                                                                                    Supervised_loss=losses_x,
                                                                                                    Consistency_loss=losses_u))

    return losses.avg, losses_x.avg, losses_u.avg, acc.avg


def validate(args,  model_student, classifier_student, val_loader, epoch):

    # switch to evaluate mode
    model_student.eval()
    classifier_student.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    with torch.no_grad():

        end = time.time()

        for batch_idx, (input, target) in enumerate(tqdm(val_loader, disable=False)):

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
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch, batch_idx + 1, len(val_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                    acc=acc))

    return losses.avg, acc.avg


def test(args, model, classifier, test_loader):

    # switch to evaluate mode
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    total_pred = []
    total_target = []
    total_pred_score = []

    with torch.no_grad():

        end = time.time()

        for batch_idx, (input, target) in enumerate(tqdm(test_loader, disable=False)):

            # Get inputs and target
            input, target = input.float(), target.long()

            # Move the variables to Cuda
            input, target = input.cuda(), target.cuda()

            # compute output ###############################
            feats = model(input)
            output = classifier(feats)
            pred_score = torch.softmax(output.detach_(), dim=-1)

            #######
            loss = F.cross_entropy(output, target, reduction='mean')

            # compute loss and accuracy
            batch_size = target.size(0)
            losses.update(loss.item(), batch_size)

            pred = torch.argmax(output, dim=1)
            acc.update(torch.sum(target == pred).item() / batch_size, batch_size)

            # Save pred, target to calculate metrics
            total_pred.append(pred)
            total_target.append(target)
            total_pred_score.append(pred_score)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print statistics and write summary every N batch
            if (batch_idx + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                    batch_idx, len(test_loader), batch_time=batch_time, loss=losses, acc=acc))

        # Pred and target for performance metrics
        final_predictions = torch.cat(total_pred).to('cpu')
        final_targets = torch.cat(total_target).to('cpu')
        final_pred_score = torch.cat(total_pred_score).to('cpu')

    return final_predictions, final_targets, final_pred_score

########
def parse_args():

    parser = argparse.ArgumentParser('Argument for Kather - Consistency training/Evaluation')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--gpu', default='0', help='GPU id to use.')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use.')
    parser.add_argument('--seed', type=int, default=42, help='seed for initializing training.')

    # model definition
    parser.add_argument('--model', type=str, default='resnet18', help='choice of network architecture.')
    parser.add_argument('--mode', type=str, default='fine-tuning', help='fine-tuning/evaluation')
    parser.add_argument('--modules_teacher', type=int, default=64,
                        help='which modules to freeze for the fine-tuned teacher model. (full-finetune(0), fine-tune only FC layer (60). Full_network(64) - Resnet18')
    parser.add_argument('--modules_student', type=int, default=60,
                        help='which modules to freeze for fine-tuning the student model. (full-finetune(0), fine-tune only FC layer (60) - Resnet18')
    parser.add_argument('--num_classes', type=int, default=9, help='# of classes.')
    parser.add_argument('--num_epoch', type=int, default=90, help='epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size.')
    parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size - 7')
    parser.add_argument('--NAug', default=7, type=int, help='No of Augmentations for strong unlabeled data')

    parser.add_argument('--lr', default=0.00001, type=float, help='learning rate. - 1e-5(Adam)')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay/weights regularizer for sgd. - 1e-4')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum for sgd, beta1 for adam.')
    parser.add_argument('--beta2', default=0.999, type=float, help=' beta2 for adam.')
    parser.add_argument('--lambda_u', default=1, type=float, help='coefficient of unlabeled loss')

    # Fine-Tuning
    parser.add_argument('--model_path_finetune', type=str,
                        default='/home/srinidhi/Research/Code/SSL_Resolution/Save_Results/Results/Kather/Fine_tune/SSL/',
                        help='path to load SSL fine-tuned model to intialize "Teacher and student network" for consistency training')
    parser.add_argument('--model_save_pth', type=str,
                        default='/home/srinidhi/Research/Code/SSL_Resolution/Save_Results/',
                        help='path to save fine-tuned model')
    parser.add_argument('--save_loss', type=str, default='/home/srinidhi/Research/Code/SSL_Resolution/Save_Results/',
                        help='path to save loss and other performance metrics')
    parser.add_argument('--resume', type=str, default='/home/srinidhi/Research/Code/SSL_Resolution/Save_Results/',
                        metavar='PATH', help='path to latest checkpoint - model.pth (default: none)')

    # Testing
    parser.add_argument('--model_path_eval', type=str,
                        default='/home/srinidhi/Research/Code/SSL_Resolution/Save_Results/Results/Kather/Fine_tune/SSL_CR/',
                        help='path to load consistency trained model')

    # Data paths
    parser.add_argument('--train_image_pth', default='/home/srinidhi/Research/Data/Kather_Multi_Class/NCT-CRC-HE-100K/')
    parser.add_argument('--test_image_pth', default='/home/srinidhi/Research/Data/Kather_Multi_Class/CRC-VAL-HE-7K/')
    parser.add_argument('--validation_split', default=0.2, type=float, help='portion of the data that will be used for validation')
    parser.add_argument('--labeled_train', default=0.1, type=float, help='portion of the train data with labels - 1(full), 0.1/0.25/0.5')

    # Tiling parameters
    parser.add_argument('--image_size', default=256, type=int, help='patch size width 256')

    args = parser.parse_args()

    return args


def main():

    # parse the args
    args = parse_args()

    # Set the data loaders (train, val, test)

    ## Kather #########

    if args.mode == 'fine-tuning':

        # Train set
        train_labeled_dataset = DatasetKather_Supervised_train(args.train_image_pth, args.image_size)
        train_unlabeled_dataset = DatasetKather_SSLtrain(args.train_image_pth, args.image_size, transform=TransformFix(args.image_size, args.NAug))

        # Validation Set
        val_dataset = DatasetKather_eval(args.train_image_pth, args.image_size)

        # train and validation split
        num_train = len(train_labeled_dataset.datalist)
        indices = list(range(num_train))
        split = int(np.floor(args.validation_split * num_train))
        np.random.shuffle(indices)
        train_idx, val_idx = indices[split:], indices[:split]
        train_labeled_idx = np.random.choice(train_idx, int(args.labeled_train * len(train_idx)))

        train_unlabeled_sampler = SubsetRandomSampler(train_idx)
        train_labeled_sampler = SubsetRandomSampler(train_labeled_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        # Data loaders
        train_labeled_loader = torch.utils.data.DataLoader(train_labeled_dataset, batch_size=args.batch_size, sampler=train_labeled_sampler,
                                                           shuffle=True if train_labeled_sampler is None else False,
                                                           num_workers=args.num_workers, pin_memory=True)

        train_unlabeled_loader = torch.utils.data.DataLoader(train_unlabeled_dataset,
                                                             batch_size=args.batch_size * args.mu,
                                                             sampler=train_unlabeled_sampler,
                                                             shuffle=True if train_unlabeled_sampler is None else False,
                                                             num_workers=args.num_workers, pin_memory=True,
                                                             drop_last=True)

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler,
                                                 shuffle=False, num_workers=args.num_workers, pin_memory=True)

        # num of samples
        n_data = len(train_labeled_sampler)
        print('number of labeled training samples: {}'.format(n_data))

        n_data = len(train_unlabeled_sampler)
        print('number of unlabeled training samples: {}'.format(n_data))

        n_data = len(val_sampler)
        print('number of validation samples: {}'.format(n_data))

    elif args.mode == 'evaluation':

        # Test set
        test_dataset = DatasetKather_eval(args.test_image_pth, args.image_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)

        # num of sample
        n_data = len(test_dataset)
        print('number of testing samples: {}'.format(n_data))

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

            ###### Intialize both teacher and student network with fine-tuned SSL model ###########

            ###### Load teacher model ###############

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

            # Freeze the Teacher model (Entire network)
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

            ################## Load student model ############################

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

            # Freeze Student model (Except last FC layer)
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

        elif args.mode == 'evaluation':

            # Load fine-tuned model (Single-GPU)
            state = torch.load(args.model_path_eval)

            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()

            for k, v in state['model_student'].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            model_student.load_state_dict(new_state_dict)

            # create new OrderedDict that does not contain `module.`
            new_state_dict_cls = OrderedDict()

            for k, v in state['classifier_student'].items():
                name = k[7:]  # remove `module.`
                new_state_dict_cls[name] = v

            classifier_student.load_state_dict(new_state_dict_cls)

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
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, list(model_student.parameters()) + list(classifier_student.parameters())), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
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

            train_losses, train_losses_x, train_losses_u, train_acc = train(args, model_teacher, model_student, classifier_teacher, classifier_student,
                                                                            train_labeled_loader, train_unlabeled_loader, optimizer, epoch)
            print('Epoch time: {:.2f} s.'.format(time.time() - time_start))

            print("==> validating the fine-tuned model...")
            val_losses, val_acc = validate(args, model_student, classifier_student, val_loader, epoch)

            # Log results
            with open(os.path.join(args.save_loss, 'fine_tuned_results.csv'), 'a') as f:
                f.write('%03d, %0.6f, %0.6f, %0.6f, %0.6f, %0.6f, %0.6f,\n' % (
                    (epoch + 1), train_losses, train_losses_x, train_losses_u, train_acc, val_losses, val_acc,))

            'adjust learning rate --- Note that step should be called after validate()'
            scheduler.step()

            # Iterative training: Use the student as a teacher
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

        elif args.mode == 'evaluation':

            print("==> testing final test data...")
            final_predictions, final_targets, final_pred_score = test(args, model_student, classifier_student, test_loader)

            final_predictions = final_predictions.numpy()
            final_targets = final_targets.numpy()
            final_pred_score = final_pred_score.numpy()

            # Kather dataset #########

            # Performance statistics of test data
            confusion_mat = multilabel_confusion_matrix(final_targets, final_predictions, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])

            tn = confusion_mat[:, 0, 0]
            tp = confusion_mat[:, 1, 1]
            fp = confusion_mat[:, 0, 1]
            fn = confusion_mat[:, 1, 0]

            se = tp / (tp + fn)
            sp = tn / (tn + fp)
            acc = (tp + tn) / (tp + tn + fp + fn)

            f1 = f1_score(final_targets, final_predictions, average='weighted')
            auc_score = roc_auc_score(final_targets, final_pred_score, multi_class='ovr')

            # Print stats
            print('Confusion Matrix', confusion_mat)
            print('Sensitivity class-wise =', se)
            print('Specificity class-wise =', sp)
            print('Accuracy class-wise =', acc)
            print('F1_score weighted =', f1)
            print('AUC_score =', auc_score)

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

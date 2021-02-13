"""
Finetuning task - Supervised fine-tuning on downstream task (Kather Dataset)
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
from util import AverageMeter, plot_confusion_matrix
from collections import OrderedDict
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler

from dataset import DatasetKather_Supervised_train, DatasetKather_eval
import models.net as net
from albumentations import Compose
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


def train(args, model, classifier, train_loader, criterion, optimizer, epoch):

    """
    Fine-tuning the pre-trained SSL model
    """

    model.train()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    total_feats = []
    total_targets = []

    end = time.time()

    for batch_idx, (input, target) in enumerate(tqdm(train_loader, disable=False)):

        # Get inputs and target
        input, target = input.float(), target.long()

        # Reshape augmented tensors
        input, target = input.reshape(-1, 3, args.image_size, args.image_size), target.reshape(-1, )

        # Move the variables to Cuda
        input, target = input.cuda(), target.cuda()

        # compute output ###############################
        feats = model(input)
        output = classifier(feats)

        ######
        loss = criterion(output, target)

        # compute gradient and do SGD step #############
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # compute loss and accuracy ####################
        batch_size = target.size(0)
        losses.update(loss.item(), batch_size)

        pred = torch.argmax(output, dim=1)
        acc.update(torch.sum(target == pred).item() / batch_size, batch_size)

        # Save features
        total_feats.append(feats)
        total_targets.append(target)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print statistics and write summary every N batch
        if (batch_idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, batch_idx + 1, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                acc=acc))

    return losses.avg, acc.avg


def validate(args, model, classifier, val_loader, criterion, epoch):

    # switch to evaluate mode
    model.eval()
    classifier.eval()

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
            feats = model(input)
            output = classifier(feats)
            loss = criterion(output, target)

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


def test(args, model, classifier, test_loader, criterion):

    # switch to evaluate mode
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    total_pred = []
    total_target = []

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

            #######
            loss = criterion(output, target)

            # compute loss and accuracy
            batch_size = target.size(0)
            losses.update(loss.item(), batch_size)

            pred = torch.argmax(output, dim=1)
            acc.update(torch.sum(target == pred).item() / batch_size, batch_size)

            # Save pred, target to calculate metrics
            total_pred.append(pred)
            total_target.append(target)

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
        final_predicitions = torch.cat(total_pred).to('cpu')
        final_targets = torch.cat(total_target).to('cpu')

    return final_predicitions, final_targets


def parse_args():

    parser = argparse.ArgumentParser('Argument for Kather: Supervised Fine-Tuning/Evaluation')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--gpu', default='0, 1', help='GPU id to use.')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use.')
    parser.add_argument('--seed', type=int, default=42, help='seed for initializing training.')

    # model definition
    parser.add_argument('--model', type=str, default='resnet18', help='choice of network architecture.')
    parser.add_argument('--mode', type=str, default='fine-tuning', help='fine-tuning/evaluation')
    parser.add_argument('--modules', type=int, default=0, help='which modules to freeze for fine-tuning the pretrained model. (full-finetune(0), fine-tune only classifier(64), layer4(45), layer3(30), layer2(15), layer1(3) - Resnet18')
    parser.add_argument('--num_classes', type=int, default=9, help='# of classes.')
    parser.add_argument('--num_epoch', type=int, default=90, help='epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size.')

    parser.add_argument('--lr', default=0.00001, type=float, help='learning rate. - 1e-5(Adam)')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay/weights regularizer for sgd. - 1e-4')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum for sgd, beta1 for adam.')
    parser.add_argument('--beta2', default=0.999, type=float, help=' beta2 for adam.')

    # Fine-tuning
    parser.add_argument('--model_path', type=str,
                        default='/home/srinidhi/Research/Data/Camelyon16/Pre_train/Camelyon16_pretrained_model.pt',
                        help='path to load self-supervised pretrained model') 'Here we load Camelyon pretrained model to perform domain adaptation from Breast Cancer to Colorectal Images. Refer to, paper for more details'
    parser.add_argument('--model_save_pth', type=str,
                        default='/home/srinidhi/Research/Code/SSL_Resolution/Save_Results/',
                        help='path to save fine-tuned model')
    parser.add_argument('--save_loss', type=str,
                        default='/home/srinidhi/Research/Code/SSL_Resolution/Save_Results/',
                        help='path to save loss and other performance metrics')
    parser.add_argument('--resume', type=str, default='/home/srinidhi/Research/Code/SSL_Resolution/Save_Results/',
                        metavar='PATH', help='path to latest checkpoint - model.pth (default: none)')

    # Testing
    parser.add_argument('--finetune_model_path', type=str,
                        default='/home/srinidhi/Research/Code/SSL_Resolution/Save_Results/',
                        help='path to load fine-tuned model for evaluation (test)')

    # Data paths
    parser.add_argument('--train_image_pth', default='/home/srinidhi/Research/Data/Kather_Multi_Class/NCT-CRC-HE-100K/')
    parser.add_argument('--test_image_pth', default='/home/srinidhi/Research/Data/Kather_Multi_Class/CRC-VAL-HE-7K/')
    parser.add_argument('--validation_split', default=0.2, type=float, help='portion of the data that will be used for validation')
    parser.add_argument('--labeled_train', default=0.1, type=float, help='portion of the train data with labels - 1(full), 0.1/0.25/0.5')

    # Tiling parameters
    parser.add_argument('--image_size', default=256, type=int, help='patch size width 256/128')

    args = parser.parse_args()

    return args


def main():

    # parse the args
    args = parse_args()

    # Set the data loaders (train, val, test)

    ## Kather #########

    if args.mode == 'fine-tuning':

        # Train set
        train_dataset = DatasetKather_Supervised_train(args.train_image_pth, args.image_size)

        # Validation Set
        val_dataset = DatasetKather_eval(args.train_image_pth, args.image_size)

        # train and validation split
        num_train = len(train_dataset.datalist)
        indices = list(range(num_train))
        split = int(np.floor(args.validation_split * num_train))
        np.random.shuffle(indices)
        train_idx, val_idx = indices[split:], indices[:split]
        train_idx = np.random.choice(train_idx, int(args.labeled_train * len(train_idx)))

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        # Data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                                   shuffle=True if train_sampler is None else False,
                                                   num_workers=args.num_workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler,
                                                 shuffle=False, num_workers=args.num_workers, pin_memory=True)

        # num of samples
        n_data = len(train_sampler)
        print('number of training samples: {}'.format(n_data))

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

    ############################################

    # set the model
    if args.model == 'resnet18':

        model = net.TripletNet_Finetune(args.model)

        if args.mode == 'fine-tuning':

            # original model saved file with DataParallel (Multi-GPU)
            state_dict = torch.load(args.model_path)

            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()

            for k, v in state_dict['model'].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            # Load pre-trained model
            print('==> loading pre-trained model')
            model.load_state_dict(new_state_dict)

            # look at the contents of the model and its parameters
            idx = 0
            for layer_name, param in model.named_parameters():
                print(layer_name, '-->', idx)
                idx += 1

            # Freezing the specific layer weights in the model and fine tune it
            for name, param in enumerate(model.named_parameters()):
                if name < args.modules:  # No of layers(modules) to be freezed
                    print("module", name, "was frozen")
                    param = param[1]
                    param.requires_grad = False
                else:
                    print("module", name, "was not frozen")
                    param = param[1]
                    param.requires_grad = True

            print('==> finetuning classification')
            classifier = net.FinetuneResNet(args.num_classes)

            # Multi-GPU
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
                classifier = torch.nn.DataParallel(classifier)

        elif args.mode == 'evaluation':

            # Load fine-tuned model
            state_dict = torch.load(args.finetune_model_path)

            # create new OrderedDict that does not contain `module.`
            new_state_dict_model = OrderedDict()
            new_state_dict_classifier = OrderedDict()

            for k, v in state_dict['model'].items():
                name = k[7:]  # remove `module.`
                new_state_dict_model[name] = v

            for k, v in state_dict['classifier'].items():
                name = k[7:]  # remove `module.`
                new_state_dict_classifier[name] = v

            # Load pre-trained model
            print('==> loading pre-trained model')
            model.load_state_dict(new_state_dict_model)

            # classifier = net.FinetuneResNet(args.num_classes)
            classifier.load_state_dict(new_state_dict_classifier)

        else:
            raise NotImplementedError('invalid training {}'.format(args.mode))

    else:
        raise NotImplementedError('model not supported {}'.format(args.model))

#########################

    # loss fn
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        classifier = classifier.cuda()
        cudnn.benchmark = True

    # Optimiser & scheduler
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters()) + list(classifier.parameters())), lr=args.lr,
                           betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

    # Training Model
    start_epoch = 1
    best_val_acc = -1

    'check resume from a checkpoint'
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model'])
            classifier.load_state_dict(checkpoint['classifier'])
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
        f.write('epoch, train_loss, train_acc, val_loss, val_acc\n')

    # Routine
    for epoch in range(start_epoch, args.num_epoch + 1):

        if args.mode == 'fine-tuning':

            print("==> fine-tuning the pretrained SSL model...")

            time_start = time.time()

            train_losses, train_acc = train(args, model, classifier, train_loader, criterion, optimizer, epoch)
            print('Epoch time: {:.2f} s.'.format(time.time() - time_start))

            print("==> validating the fine-tuned model...")
            val_losses, val_acc = validate(args, model, classifier, val_loader, criterion, epoch)

            # Log results
            with open(os.path.join(args.save_loss, 'fine_tuned_results.csv'), 'a') as f:
                f.write('%03d,%0.6f,%0.6f,%0.6f,%0.6f,\n' % ((epoch + 1), train_losses, train_acc, val_losses, val_acc))

            'adjust learning rate --- Note that step should be called after validate()'
            scheduler.step()

            # Save model every 10 epochs
            if epoch % args.save_freq == 0:
                print('==> Saving...')
                state = {
                    'args': args,
                    'model': model.state_dict(),
                    'classifier': classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_losses,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'val_loss': val_losses
                }
                torch.save(state, '{}/fine_tuned_model_{}.pt'.format(args.model_save_pth, epoch))

            # Save model for the best val
            if val_acc > best_val_acc:
                print('==> Saving...')
                state = {
                    'args': args,
                    'model': model.state_dict(),
                    'classifier': classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_losses,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'val_loss': val_losses
                }
                torch.save(state, '{}/best_fine_tuned_model_{}.pt'.format(args.model_save_pth, epoch))
                best_val_acc = val_acc

                # help release GPU memory
                del state

            torch.cuda.empty_cache()

        elif args.mode == 'evaluation':

            print("==> testing final test data...")
            final_predicitions, final_targets = test(args, model, classifier, test_loader, criterion)

            final_predicitions = final_predicitions.numpy()
            final_targets = final_targets.numpy()

            # Kather dataset #########

            # Performance statistics of test data
            confusion_mat = multilabel_confusion_matrix(final_targets, final_predicitions, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])

            tn = confusion_mat[:, 0, 0]
            tp = confusion_mat[:, 1, 1]
            fp = confusion_mat[:, 0, 1]
            fn = confusion_mat[:, 1, 0]

            se = tp / (tp + fn)
            sp = tn / (tn + fp)
            acc = (tp + tn) / (tp + tn + fp + fn)

            f1 = f1_score(final_targets, final_predicitions, average='weighted')

            # Print stats
            print('Confusion Matrix', confusion_mat)
            print('Sensitivity class-wise =', se)
            print('Specificity class-wise =', sp)
            print('Accuracy class-wise =', acc)
            print('F1_score weighted =', f1)

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

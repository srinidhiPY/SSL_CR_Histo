"""
Train Multi-Resolution Self-Supervision - Camelyon16
"""
from __future__ import print_function
import argparse
import os
import time
import random
from tqdm import tqdm
import torch.backends.cudnn as cudnn

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import torch.optim as optim
import torch.nn as nn
from util import AverageMeter
from torchvision import transforms

from dataset import DatasetWSIs
import models.net as net
from models.optimiser.RAdam.lookahead import Lookahead
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def train(args, model, classifier, train_loader, criterion, optimizer, epoch):

    # Switch to train mode
    model.train()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    total_feats = []
    total_targets = []

    end = time.time()

    for batch_idx, (input1, input2, input3, target) in enumerate(tqdm(train_loader, disable=False)):

        # Get inputs and target
        input1, input2, input3, target = input1.float(), input2.float(), input3.float(), target.long()

        # Reshape augmented tensors
        input1, input2, input3, target = input1.reshape(-1, 3, args.tile_h, args.tile_w), input2.reshape(-1, 3, args.tile_h, args.tile_w), input3.reshape(-1, 3, args.tile_h, args.tile_w), target.view(-1, 1).reshape(-1, )

        # Move the variables to Cuda
        input1, input2, input3, target = input1.cuda(), input2.cuda(), input3.cuda(), target.cuda()

        # compute output ###############################
        feats = model(input1, input2, input3)
        output = classifier(feats)
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
        torch.cuda.synchronize()
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

        final_feats = torch.cat(total_feats).detach()
        final_targets = torch.cat(total_targets).detach()

    return losses.avg, acc.avg, final_feats, final_targets


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

        for batch_idx, (input1, input2, input3, target) in enumerate(tqdm(val_loader, disable=False)):

            # Get inputs and target
            input1, input2, input3, target = input1.float(), input2.float(), input3.float(), target.long()

            # Reshape augmented tensors
            input1, input2, input3, target = input1.reshape(-1, 3, args.tile_h, args.tile_w), input2.reshape(-1, 3, args.tile_h, args.tile_w), input3.reshape(-1, 3, args.tile_h, args.tile_w), target.view(-1, 1).reshape(-1, )

            # Move the variables to Cuda
            input1, input2, input3, target = input1.cuda(), input2.cuda(), input3.cuda(), target.cuda()

            # compute output ###############################
            feats = model(input1, input2, input3)
            output = classifier(feats)
            loss = criterion(output, target)

            # compute loss and accuracy ####################
            batch_size = target.size(0)
            losses.update(loss.item(), batch_size)
            pred = torch.argmax(output, dim=1)

            acc.update(torch.sum(target == pred).item() / batch_size, batch_size)

            # measure elapsed time
            torch.cuda.synchronize()
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


def parse_args():

    parser = argparse.ArgumentParser('Argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--gpu', default='0, 1', help='GPU id to use.')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use.')
    parser.add_argument('--seed', type=int, default=42, help='seed for initializing training.')

    # model definition
    parser.add_argument('--model', type=str, default='resnet18', help='choice of network architecture - resnet18/resnet50.')
    parser.add_argument('--num_classes', type=int, default=6, help='# of classes.')
    parser.add_argument('--num_epoch', type=int, default=250, help='epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size.')

    parser.add_argument('--lr', default=0.01, type=float, help='learning rate - 0.01(Lookahead+SGD with Nestrov)')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay/weights regularizer for sgd. - 1e-4')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum for sgd, beta1 for adam.')
    parser.add_argument('--beta2', default=0.999, type=float, help=' beta2 for adam.')

    # Data paths
    parser.add_argument('--train_image_pth', default='/home/srinidhi/Research/Data/Camelyon16/Pre-train/WSI/',
                        help='path to train images WSIs')
    parser.add_argument('--output_pth', default='/home/srinidhi/Research/Data/Camelyon16/Pre-train/WSI/output/',
                        help='path to save tiles for visualization')
    parser.add_argument('--model_save_pth', type=str,
                        default='/home/srinidhi/Research/Code/Git_Hub/SSL_CR_Histo/Save_Results/',
                        help='path to save model')
    parser.add_argument('--save_loss', type=str,
                        default='/home/srinidhi/Research/Code/Git_Hub/SSL_CR_Histo/Save_Results/',
                        help='path to save loss')
    parser.add_argument('--resume', default='/home/srinidhi/Research/Code/Git_Hub/SSL_CR_Histo/Save_Results/', type=str,
                        metavar='PATH', help='path to latest checkpoint - model.pth (default: none)')

    # WSI tiling parameters
    parser.add_argument('--tile_w', default=256, type=int, help='patch size width')
    parser.add_argument('--tile_h', default=256, type=int, help='patch size height')
    parser.add_argument('--tile_stride_w', default=512, type=int, help='stride width dx @lowest resolution')
    parser.add_argument('--tile_stride_h', default=512, type=int, help='stride height dy @lowest resolution')
    parser.add_argument('--lwst_level_idx', default=5, type=int, help='Select Lowest level for patch indexing')

    args = parser.parse_args()

    return args

##############
def main():

    # parse the args
    args = parse_args()

    # Set the loader
    train_dataset = DatasetWSIs(args.train_image_pth, args.output_pth, args.tile_h, args.tile_w, args.tile_stride_h,
                                args.tile_stride_w, args.lwst_level_idx)

    # Train and validation split
    train_dataset, val_dataset = random_split(train_dataset, (len(train_dataset) - 10000, 10000))

    # train/val loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    # num of samples
    n_data = len(train_dataset)
    print('number of training samples: {}'.format(n_data))

    n_data = len(val_dataset)
    print('number of validation samples: {}'.format(n_data))

    # set the model
    model = net.TripletNet(args.model)
    in_features = 256
    classifier = net.Classifier(in_features * 3, args.num_classes)

    # Multi-GPU
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        classifier = torch.nn.DataParallel(classifier)

    # loss fn
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # Optimiser & scheduler
    optimizer = optim.SGD(list(model.parameters()) + list(classifier.parameters()), lr=args.lr,
                          momentum=args.beta1, weight_decay=args.weight_decay, nesterov=True)
    scheduler = Lookahead(optimizer, la_steps=5, la_alpha=0.5)
    #######

    # Training Model
    start_epoch = 1
    prev_best_val_loss = float('inf')

    'check resume from a checkpoint'
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Start log (writing into XL sheet)
    with open(os.path.join(args.save_loss, 'train_results.csv'), 'w') as f:
        f.write('epoch, train_loss, train_acc, val_loss, val_acc\n')

    # Routine
    for epoch in range(start_epoch, args.num_epoch + 1):

        print("==> training...")

        time_start = time.time()

        train_losses, train_acc, final_feats, final_targets = train(args, model, classifier, train_loader, criterion, optimizer, epoch)
        print('Epoch time: {:.2f} s.'.format(time.time() - time_start))

        print("==> validation...")
        val_losses, val_acc = validate(args, model, classifier, val_loader, criterion, epoch)

        # Log results
        with open(os.path.join(args.save_loss, 'train_results.csv'), 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.6f,%0.6f,\n' % ((epoch + 1), train_losses, train_acc, val_losses, val_acc))

        'adjust learning rate --- Note that step should be called after validate()'
        scheduler.step()

        # Save model every 10 epochs
        if (epoch % args.save_freq == 0):
            print('==> Saving...')
            state = {
                'args': args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': train_losses,
                'train_acc': train_acc,
            }
            torch.save(state, '{}/model_{}.pt'.format(args.model_save_pth, epoch))

        # Save model for the best val
        if (val_losses < prev_best_val_loss) & (epoch > 80):
            print('==> Saving...')
            state = {
                'args': args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': train_losses,
                'train_acc': train_acc,
            }
            torch.save(state, '{}/best_model_{}.pt'.format(args.model_save_pth, epoch))
            prev_best_val_loss = val_losses

            # T-sne Visualization
            final_feats = final_feats.to('cpu')
            final_feats = final_feats.numpy()
            final_targets = final_targets.to('cpu')
            final_targets = final_targets.numpy()
            np.save('{}/best_pre_trained_feats_{}'.format(args.model_save_pth, epoch), final_feats)
            np.save('{}/best_pre_trained_targets_{}'.format(args.model_save_pth, epoch), final_targets)

            # T-sne Visualization
            tsne = TSNE()
            Y = tsne.fit_transform(final_feats)
            plt.figure(figsize=(8, 8))
            colors = 'r', 'g', 'b', 'c', 'm', 'y'
            target_names = ['0', '1', '2', '3', '4', '5']
            target_ids = range(len(target_names))
            for i, c, label in zip(target_ids, colors, target_names):
                plt.scatter(Y[final_targets==i, 0], Y[final_targets==i, 1], c=c, label=label)
            plt.legend()
            plt.savefig('{}/best_tsne_feats_{}.png'.format(args.model_save_pth, epoch), dpi=300)

            # help release GPU memory
            del state

        torch.cuda.empty_cache()


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

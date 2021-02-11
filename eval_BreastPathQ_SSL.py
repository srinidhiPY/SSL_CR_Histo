"""
Finetuning task - Supervised fine-tuning on downstream task (BreastPathQ)
"""
import argparse
import os
import time
import random
import numpy as np
from PIL import Image
import cv2
import pingouin as pg
import statsmodels.api as sm
import pandas as pd
from tqdm import tqdm
import torch.backends.cudnn as cudnn

import torch
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
from util import AverageMeter, plot_confusion_matrix
from collections import OrderedDict
from torchvision import transforms, datasets

from dataset import DatasetBreastPathQ_Supervised_train, DatasetBreastPathQ_eval, DatasetBreastPathQ_SSLtrain
import models.net as net
from albumentations import Compose
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler


#############
def train(args, model, classifier, train_loader, criterion, optimizer, epoch):

    """
    Fine-tuning the pre-trained SSL model
    """

    model.train()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    total_feats = []
    total_targets = []

    end = time.time()

    for batch_idx, (input1, target) in enumerate(tqdm(train_loader, disable=False)):

        # Get inputs and target
        input1, target = input1.float(), target.float()

        # Reshape augmented tensors
        input1, target = input1.reshape(-1, 3, args.image_size, args.image_size), target.reshape(-1, )

        # Move the variables to Cuda
        input1, target = input1.cuda(), target.cuda()

        # compute output ###############################
        feats = model(input1)
        output = classifier(feats)

        # BreastPathQ dataset
        output = output.view(-1, 1).reshape(-1, )

        ######
        loss = criterion(output, target)

        # compute gradient and do SGD step #############
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # compute loss and accuracy ####################
        batch_size = target.size(0)
        losses.update(loss.item(), batch_size)

        # Save features
        total_feats.append(feats)
        total_targets.append(target)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # BreastPathQ #######

        # print statistics and write summary every N batch
        if (batch_idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, batch_idx + 1, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses))

        final_feats = torch.cat(total_feats).detach()
        final_targets = torch.cat(total_targets).detach()

    return losses.avg, final_feats, final_targets


def validate(args, model, classifier, val_loader, criterion, epoch):

    # switch to evaluate mode
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    with torch.no_grad():

        end = time.time()

        for batch_idx, (input1, target) in enumerate(tqdm(val_loader, disable=False)):

            # Get inputs and target
            input1, target = input1.float(), target.long()

            # Move the variables to Cuda
            input1, target = input1.cuda(), target.cuda()

            # compute output ###############################
            feats = model(input1)
            output = classifier(feats)
            loss = criterion(output, target.view(-1, 1))

            # compute loss and accuracy ####################
            batch_size = target.size(0)
            losses.update(loss.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print statistics and write summary every N batch
            if (batch_idx + 1) % args.print_freq == 0:
                print('Val: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})'.format(epoch, batch_idx + 1, len(val_loader),
                                                                    batch_time=batch_time, data_time=data_time, loss=losses))

    return losses.avg


def test(args, model, classifier, criterion, test_loader):

    # switch to evaluate mode
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()

    total_feats = []
    total_output = []

    total_targetA = []
    total_targetB = []

    with torch.no_grad():

        end = time.time()

        for batch_idx, (input, targetA, targetB) in enumerate(tqdm(test_loader, disable=False)):

            # Get inputs and target
            input, targetA, targetB = input.float(), targetA.float(), targetB.float()

            # Move the variables to Cuda
            input, targetA, targetB = input.cuda(), targetA.cuda(), targetB.cuda()

            # compute output ###############################
            feats = model(input)
            output = classifier(feats)

            #######
            loss = criterion(output, targetA.view(-1, 1))

            # compute loss and accuracy
            batch_size = targetA.size(0)
            losses.update(loss.item(), batch_size)

            # Save pred, target to calculate metrics
            output = output.view(-1, 1).reshape(-1, )
            total_output.append(output)
            total_feats.append(feats)

            total_targetA.append(targetA)
            total_targetB.append(targetB)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print statistics and write summary every N batch
            if (batch_idx + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                    batch_idx, len(test_loader), batch_time=batch_time, loss=losses))

        # Pred and target for performance metrics
        final_outputs = torch.cat(total_output).to('cpu')
        final_feats = torch.cat(total_feats).to('cpu')

        final_targetsA = torch.cat(total_targetA).to('cpu')
        final_targetsB = torch.cat(total_targetB).to('cpu')

    return final_outputs, final_feats, final_targetsA, final_targetsB


def parse_args():

    parser = argparse.ArgumentParser('Argument for BreastPathQ: Supervised Fine-Tuning/Evaluation')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--gpu', default='0', help='GPU id to use.')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use.')
    parser.add_argument('--seed', type=int, default=42, help='seed for initializing training.')

    # model definition
    parser.add_argument('--model', type=str, default='resnet18', help='choice of network architecture.')
    parser.add_argument('--mode', type=str, default='fine-tuning', help='fine-tuning/evaluation')
    parser.add_argument('--modules', type=int, default=0, help='which modules to freeze for fine-tuning the pretrained model. (full-finetune(0), fine-tune only FC layer (60) - Resnet18')
    parser.add_argument('--num_classes', type=int, default=1, help='# of classes.')
    parser.add_argument('--num_epoch', type=int, default=90, help='epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size.')

    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate. - 1e-4(Adam)')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay/weights regularizer for sgd. - 1e-4')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum for sgd, beta1 for adam.')
    parser.add_argument('--beta2', default=0.999, type=float, help=' beta2 for adam.')

    # Fine-tuning
    parser.add_argument('--model_path', type=str,
                        default='/home/srinidhi/Research/Data/BreastPathQ/BreastPathQ_pretrained_model.pt',
                        help='path to load self-supervised pretrained model')
    parser.add_argument('--model_save_pth', type=str,
                        default='/home/srinidhi/Research/Code/SSL_Resolution/Save_Results/',
                        help='path to save fine-tuned model')
    parser.add_argument('--save_loss', type=str,
                        default='/home/srinidhi/Research/Code/SSL_Resolution/Save_Results/',
                        help='path to save loss and other performance metrics')

    # Testing
    parser.add_argument('--finetune_model_path', type=str,
                        default='/home/srinidhi/Research/Code/SSL_Resolution/Save_Results/',
                        help='path to load fine-tuned model for evaluation (test)')

    # Data paths
    parser.add_argument('--train_image_pth', default='/home/srinidhi/Research/Data/Cellularity/Tumor_Cellularity_Compare/TrainSet/')
    parser.add_argument('--test_image_pth', default='/home/srinidhi/Research/Data/Cellularity/Tumor_Cellularity_Compare/')
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

    ### BreastPathQ #######

    if args.mode == 'fine-tuning':

        # Train set
        train_transforms = Compose([])
        train_dataset = DatasetBreastPathQ_Supervised_train(args.train_image_pth, args.image_size, transform=train_transforms)

        # Validation set
        transform_val = transforms.Compose([transforms.Resize(size=args.image_size)])
        val_dataset = DatasetBreastPathQ_SSLtrain(args.train_image_pth, transform=transform_val)

        # train and validation split
        num_train = len(train_dataset.datalist)
        indices = list(range(num_train))
        split = int(np.floor(args.validation_split * num_train))
        np.random.shuffle(indices)
        train_idx, val_idx = indices[split:], indices[:split]
        train_idx = np.random.choice(train_idx, int(args.labeled_train * len(train_idx)))

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        # loader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=True if train_sampler is None else False,
                                                   num_workers=args.num_workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        # num of samples
        n_data = len(train_dataset)
        print('number of training samples: {}'.format(n_data))

        n_data = len(val_sampler)
        print('number of validation samples: {}'.format(n_data))

    elif args.mode == 'evaluation':

        # Test set
        test_transforms = transforms.Compose([transforms.Resize(size=args.image_size)])
        test_dataset = DatasetBreastPathQ_eval(args.test_image_pth, args.image_size, test_transforms)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)

        # num of samples
        n_data = len(test_dataset)
        print('number of testing samples: {}'.format(n_data))

    else:
        raise NotImplementedError('invalid mode {}'.format(args.mode))

    ########################################

    # set the model
    if args.model == 'resnet18':

        model = net.TripletNet_Finetune(args.model)
        classifier = net.FinetuneResNet(args.num_classes)

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

        elif args.mode == 'evaluation':

            # Load fine-tuned model
            state = torch.load(args.finetune_model_path)
            model.load_state_dict(state['model'])

            # Load fine-tuned classifier
            classifier.load_state_dict(state['classifier'])

        else:
            raise NotImplementedError('invalid training {}'.format(args.mode))

    else:
        raise NotImplementedError('model not supported {}'.format(args.model))

    criterion = nn.MSELoss()

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
    prev_best_val_loss = float('inf')

    # Start log (writing into XL sheet)
    with open(os.path.join(args.save_loss, 'fine_tuned_results.csv'), 'w') as f:
        f.write('epoch, train_loss, val_loss\n')

    # Routine
    for epoch in range(start_epoch, args.num_epoch + 1):

        if args.mode == 'fine-tuning':

            print("==> fine-tuning the pretrained SSL model...")

            time_start = time.time()

            train_losses, final_feats, final_targets = train(args, model, classifier, train_loader, criterion, optimizer, epoch)
            print('Epoch time: {:.2f} s.'.format(time.time() - time_start))

            print("==> validating the fine-tuned model...")
            val_losses = validate(args, model, classifier, val_loader, criterion, epoch)

            # Log results
            with open(os.path.join(args.save_loss, 'fine_tuned_results.csv'), 'a') as f:
                f.write('%03d,%0.6f,%0.6f,\n' % ((epoch + 1), train_losses, val_losses))

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
                }
                torch.save(state, '{}/fine_tuned_model_{}.pt'.format(args.model_save_pth, epoch))

            # Save model for the best val
            if (val_losses < prev_best_val_loss) & (epoch > 1):
                print('==> Saving...')
                state = {
                    'args': args,
                    'model': model.state_dict(),
                    'classifier': classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_losses,
                }
                torch.save(state, '{}/best_fine_tuned_model_{}.pt'.format(args.model_save_pth, epoch))
                prev_best_val_loss = val_losses

                # help release GPU memory
                del state

            torch.cuda.empty_cache()

        elif args.mode == 'evaluation':

            print("==> testing final test data...")
            final_predicitions, final_feats, final_targetsA, final_targetsB = test(args, model, classifier, criterion, test_loader)

            final_predicitions = final_predicitions.numpy()
            final_targetsA = final_targetsA.numpy()
            final_targetsB = final_targetsB.numpy()

            # BreastPathQ dataset #######
            d = {'targets': np.hstack(
                [np.arange(1, len(final_predicitions) + 1, 1), np.arange(1, len(final_predicitions) + 1, 1)]),
                'raters': np.hstack([np.tile(np.array(['M']), len(final_predicitions)),
                                     np.tile(np.array(['A']), len(final_predicitions))]),
                'scores': np.hstack([final_predicitions, final_targetsA])}
            df = pd.DataFrame(data=d)
            iccA = pg.intraclass_corr(data=df, targets='targets', raters='raters', ratings='scores')
            iccA.to_csv(os.path.join(args.save_loss, 'BreastPathQ_ICC_Eval_2way_MA.csv'))
            print(iccA)

            d = {'targets': np.hstack(
                [np.arange(1, len(final_predicitions) + 1, 1), np.arange(1, len(final_predicitions) + 1, 1)]),
                'raters': np.hstack([np.tile(np.array(['M']), len(final_predicitions)),
                                     np.tile(np.array(['B']), len(final_predicitions))]),
                'scores': np.hstack([final_predicitions, final_targetsB])}
            df = pd.DataFrame(data=d)
            iccB = pg.intraclass_corr(data=df, targets='targets', raters='raters', ratings='scores')
            iccB.to_csv(os.path.join(args.save_loss, 'BreastPathQ_ICC_Eval_2way_MB.csv'))
            print(iccB)

            d = {'targets': np.hstack(
                [np.arange(1, len(final_targetsA) + 1, 1), np.arange(1, len(final_targetsB) + 1, 1)]),
                'raters': np.hstack(
                    [np.tile(np.array(['A']), len(final_targetsA)), np.tile(np.array(['B']), len(final_targetsB))]),
                'scores': np.hstack([final_targetsA, final_targetsB])}
            df = pd.DataFrame(data=d)
            iccC = pg.intraclass_corr(data=df, targets='targets', raters='raters', ratings='scores')
            iccC.to_csv(os.path.join(args.save_loss, 'BreastPathQ_ICC_Eval_2way_AB.csv'))
            print(iccC)

            # Plots
            fig, ax = plt.subplots()  # P1 vs automated
            ax.scatter(final_targetsA, final_predicitions, edgecolors=(0, 0, 0))
            ax.plot([final_targetsA.min(), final_targetsA.max()], [final_targetsA.min(), final_targetsA.max()], 'k--',
                    lw=2)
            ax.set_xlabel('Pathologist1')
            ax.set_ylabel('Automated Method')
            plt.savefig(os.path.join(args.save_loss, 'BreastPathQ_Eval_2way_MA_plot.png'), dpi=300)
            plt.show()

            fig, ax = plt.subplots()  # P2 vs automated
            ax.scatter(final_targetsB, final_predicitions, edgecolors=(0, 0, 0))
            ax.plot([final_targetsB.min(), final_targetsB.max()], [final_targetsB.min(), final_targetsB.max()], 'k--',
                    lw=2)
            ax.set_xlabel('Pathologist2')
            ax.set_ylabel('Automated Method')
            plt.savefig(os.path.join(args.save_loss, 'BreastPathQ_Eval_2way_MB_plot.png'), dpi=300)
            plt.show()

            fig, ax = plt.subplots()  # P1 vs P2
            ax.scatter(final_targetsA, final_targetsB, edgecolors=(0, 0, 0))
            ax.plot([final_targetsA.min(), final_targetsA.max()], [final_targetsA.min(), final_targetsA.max()], 'k--',
                    lw=2)
            ax.set_xlabel('Pathologist1')
            ax.set_ylabel('Pathologist2')
            plt.savefig(os.path.join(args.save_loss, 'BreastPathQ_Eval_2way_AB_plot.png'), dpi=300)
            plt.show()

            # Bland altman plot
            fig, ax = plt.subplots(1, figsize=(8, 8))
            sm.graphics.mean_diff_plot(final_targetsA, final_predicitions, ax=ax)
            plt.savefig(os.path.join(args.save_loss, 'BDPlot_Eval_2way_MA_plot.png'), dpi=300)
            plt.show()

            fig, ax = plt.subplots(1, figsize=(8, 8))
            sm.graphics.mean_diff_plot(final_targetsB, final_predicitions, ax=ax)
            plt.savefig(os.path.join(args.save_loss, 'BDPlot_Eval_2way_MB_plot.png'), dpi=300)
            plt.show()

            fig, ax = plt.subplots(1, figsize=(8, 8))
            sm.graphics.mean_diff_plot(final_targetsA, final_targetsB, ax=ax)
            plt.savefig(os.path.join(args.save_loss, 'BDPlot_Eval_2way_AB_plot.png'), dpi=300)
            plt.show()

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

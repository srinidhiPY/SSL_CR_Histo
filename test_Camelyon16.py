"""
Test script for Camelyon16 Probability map generation
"""
import argparse
import os
import time
import random
import numpy as np
from PIL import Image
import cv2
import glob
from tqdm import tqdm
import torch.backends.cudnn as cudnn

import torch
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
from util import AverageMeter
from collections import OrderedDict
from torchvision import transforms, datasets
import torch.nn.functional as F

from dataset import DatasetCamelyon16_test
import models.net as net
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#####
def test(args, model, classifier, test_loader):

    # switch to evaluate mode
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()

    with torch.no_grad():

        end = time.time()
        probs_map = np.zeros(test_loader.dataset.mask.shape)

        for batch_idx, (input, x_mask, y_mask) in enumerate(tqdm(test_loader, disable=False)):

            # Get inputs and target
            input = input.cuda()
            x_mask = x_mask.data.numpy()
            y_mask = y_mask.data.numpy()

            # compute output ############
            feats = model(input)
            output = classifier(feats)

            #######
            probs = torch.softmax(output, dim=1).cpu()
            probs = probs[:, -1]  # second column 'tumor'
            probs = probs.data.numpy()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print statistics and write summary every N batch
            if (batch_idx + 1) % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(batch_idx, len(test_loader), batch_time=batch_time))

            probs_map[x_mask, y_mask] = probs

    return probs_map


def parse_args():

    parser = argparse.ArgumentParser('Argument for Camelyon16 test predictions')

    parser.add_argument('--gpu', default='0', help='GPU id to use.')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use.')
    parser.add_argument('--seed', type=int, default=42, help='seed for initializing training.')

    # model definition
    parser.add_argument('--model', type=str, default='resnet18', help='choice of network architecture.')
    parser.add_argument('--num_classes', type=int, default=2, help='# of classes.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size.')

    parser.add_argument('--finetune_model_path', type=str, default='/home/srinidhi/projects/Camelyon16/',
                        help='path to load fine-tuned model for evaluation/test')

    # Data paths
    parser.add_argument('--test_image_pth', default='/home/srinidhi/projects/Camelyon16/testing/Images/')
    parser.add_argument('--test_mask_pth', default='/home/srinidhi/projects/Camelyon16/test_mask/')
    parser.add_argument('--probs_map_path', default='/home/srinidhi/projects/Camelyon16/Results/SSL/')

    # Tiling parameters
    parser.add_argument('--image_size', default=256, type=int, help='patch size width 256')

    args = parser.parse_args()

    return args

########
def main():

    # parse the args
    args = parse_args()

    # set the model
    if args.model == 'resnet18':

        model = net.TripletNet_Finetune(args.model)

        # original model saved file with DataParallel (Multi-GPU)
        state_dict = torch.load(args.finetune_model_path)

        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()

        for k, v in state_dict['model'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        # Load pre-trained model
        print('==> loading pre-trained model')
        model.load_state_dict(new_state_dict)

        print('==> classification')
        classifier = net.FinetuneResNet(args.num_classes)

    else:
        raise NotImplementedError('model not supported {}'.format(args.model))

    # Load model to CUDA
    if torch.cuda.is_available():
        model = model.cuda()
        classifier = classifier.cuda()
        cudnn.benchmark = True

    ####### Camelyon16 Evaluation Script ########################

    wsipaths = []
    maskpaths = []

    for file_ext in ['tif', 'svs', 'npy']:
        wsipaths = wsipaths + glob.glob('{}/*.{}'.format(args.test_image_pth, file_ext))
        maskpaths = maskpaths + glob.glob('{}/*.{}'.format(args.test_mask_pth, file_ext))
        wsipaths, maskpaths = sorted(wsipaths), sorted(maskpaths)

    for file_ID in range(len(wsipaths)):

        wsi_pth = wsipaths[file_ID]
        mask_pth = maskpaths[file_ID]

        wsi_id = str(os.path.split(wsi_pth)[-1])
        wsi_id = os.path.splitext(wsi_id)[0]

        # Test set
        test_dataset = DatasetCamelyon16_test(wsi_pth, mask_pth, args.image_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        #####
        n_data = len(test_dataset)
        print('number of testing samples: {}'.format(n_data))

        #################

        # Testing Model
        print("==> testing final test data...")
        probs_map = test(args, model, classifier, test_loader)

        # Save predictions
        np.save(os.path.join(args.probs_map_path, wsi_id), probs_map)

        probs_map = np.transpose(probs_map)
        predicted_img = Image.fromarray(np.uint8(probs_map * 255))
        predicted_img.save(os.path.join(args.probs_map_path, wsi_id + "." + 'png'), "PNG")
        predicted_img.close()

        # Save Heat-map
        cmapper = cm.get_cmap('jet')
        probs_heatmap = Image.fromarray(np.uint8(cmapper(np.clip(probs_map, 0, 1)) * 255))
        probs_heatmap.save(os.path.join(args.probs_map_path, wsi_id + "_" + 'heatmap' + "." + 'png'), "PNG")
        probs_heatmap.close()

        # Plot heatmap-colorbar
        plt.imshow(probs_map, cmap='jet', interpolation='nearest')
        plt.colorbar()
        plt.clim(0.00, 1.00)
        plt.axis('off')
        plt.savefig(os.path.join(args.probs_map_path, wsi_id + "_" + 'heatmap_bar' + "." + 'png'), bbox_inches='tight', dpi=300)
        plt.clf()

        del probs_map, cmapper, predicted_img, probs_heatmap


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

import numpy as np
import torch
import os
import openslide
import glob
import random
import h5py
from os.path import exists
import copy
from tqdm import tqdm
from skimage import color
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps
from util import isforeground
from models.randaugment import RandAugment

###########################################################################

# Sequence ordering of multi-resolution tiles from WSIs - Self-Supervised Trick

def sorted_sequence(all_image_tiles_hr, all_image_tiles_lr1, all_image_tiles_lr2):

    Data_1 = []
    Data_2 = []
    Data_3 = []
    Class_labels = []

    for index in range(len(all_image_tiles_hr)):

        tuple_images = [all_image_tiles_hr[index], all_image_tiles_lr1[index],
                        all_image_tiles_lr2[index]]  # Set of tuple images - HR, LR1, LR2

        # Order prediction
        sorting_orders = [[0, 1, 2], [0, 2, 1], [1, 2, 0], [1, 0, 2], [2, 0, 1],
                          [2, 1, 0]]  # 3! = 6 combination of shuffled sequences
        labels = [0, 1, 2, 3, 4, 5]  # 3! = 6 class multi-class classification

        count = 0
        for i in range(len(sorting_orders)):

            # Select sorting sequence
            img_ordering = sorting_orders[i]

            # Ordered image set
            img_1 = tuple_images[img_ordering[0]]
            img_2 = tuple_images[img_ordering[1]]
            img_3 = tuple_images[img_ordering[2]]

            # Class label
            Class_ids = np.array([labels[i]])

            Data_1.append(img_1)
            Data_2.append(img_2)
            Data_3.append(img_3)
            Class_labels.append([Class_ids])

            count = count + 1

    # Concatenate
    Data_1 = np.stack(Data_1, axis=0).astype('uint8')
    Data_2 = np.stack(Data_2, axis=0).astype('uint8')
    Data_3 = np.stack(Data_3, axis=0).astype('uint8')
    Class_labels = np.stack(Class_labels, axis=0).astype('uint8')

    return Data_1, Data_2, Data_3, Class_labels


#################
class TensorDataset_Transform(Dataset):

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)

        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        Data_1 = self.tensors[0][index]
        Data_2 = self.tensors[1][index]
        Data_3 = self.tensors[2][index]

        Class_labels = self.tensors[3][index]

        if self.transform:
            # Convert tensors to numpy array
            Data_1 = Data_1.numpy()
            Data_2 = Data_2.numpy()
            Data_3 = Data_3.numpy()
            Class_labels = Class_labels.numpy()

            # Augmentations
            Data_1 = self.transform(Data_1)
            Data_2 = self.transform(Data_2)
            Data_3 = self.transform(Data_3)

            # Numpy to torch
            Data_1 = torch.from_numpy(Data_1)
            Data_2 = torch.from_numpy(Data_2)
            Data_3 = torch.from_numpy(Data_3)
            Class_labels = torch.from_numpy(Class_labels)

            # Change dimension to N x 1
            Class_labels = Class_labels.view(-1, 1).reshape(-1, )

            # Change Tensor Dimension to N x C x H x W
            Data_1 = Data_1.permute(2, 0, 1)
            Data_2 = Data_2.permute(2, 0, 1)
            Data_3 = Data_3.permute(2, 0, 1)

        return Data_1, Data_2, Data_3, Class_labels

    def __len__(self):
        return self.tensors[0].size(0)


# Read images and extract tiles
def DatasetWSIs(image_pth, output_pth, tile_h, tile_w, tile_stride_h, tile_stride_w, lwst_level_idx, NAug, Magn):

    # Read slide
    si = Vectorize_WSIs(image_pth, output_pth, tile_h, tile_w, tile_stride_h, tile_stride_w, lwst_level_idx)

    # Generate tiles
    all_image_tiles_hr, all_image_tiles_lr1, all_image_tiles_lr2 = si.tiles_array()

    # Sort data according to ordered sequence
    Data_1, Data_2, Data_3, Class_labels = sorted_sequence(all_image_tiles_hr, all_image_tiles_lr1, all_image_tiles_lr2)

    Data_1 = torch.from_numpy(Data_1)
    Data_2 = torch.from_numpy(Data_2)
    Data_3 = torch.from_numpy(Data_3)
    Class_labels = torch.from_numpy(Class_labels)

    shuffle_idx = torch.randperm(len(Data_1))

    Data_1 = Data_1[shuffle_idx, :, :, :]
    Data_2 = Data_2[shuffle_idx, :, :, :]
    Data_3 = Data_3[shuffle_idx, :, :, :]
    Class_labels = Class_labels[shuffle_idx, :]

    # Data transforms
    train_transforms = transforms.Compose([RandAugment(NAug, Magn)])

    # Tensor dataset with transforms
    dataset = TensorDataset_Transform(tensors=(Data_1, Data_2, Data_3, Class_labels),
                                      transform=train_transforms)
    return dataset


#########
class Vectorize_WSIs:
    """" WSI dataset """

    def __init__(self, image_pth, output_pth, tile_h, tile_w, tile_stride_h, tile_stride_w, lwst_level_idx):

        """
        Args:
            image_pth (str): path to wsi.
            output_pth (str): path to save tiles
            tile_h (int): tile height
            tile_w (int): tile width
            tile_stride_h (int): stride height
            tile_stride_w (int): stride width
            lwst_level_idx (int): lowest level for patch indexing
        """

        self.image_path = image_pth
        self.output_path = output_pth
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.tile_stride_h = tile_stride_h
        self.tile_stride_w = tile_stride_w
        self.lwst_level_idx = lwst_level_idx
        self.lr_level_2, self.lr_level_1, self.hr_level = 2, 1, 0  # (2, 1, 0) assuming 20x is level 0 (BreastPathQ dataset); (2, 1, 0) 20x is level 1, 40X is level 0 for Camelyon16

    def tiles_array(self):

        # Check image
        if not exists(self.image_path):
            raise Exception('WSI file does not exist in: %s' % str(self.image_path))

        wsipaths = []
        for file_ext in ['tif', 'svs']:
            wsipaths = wsipaths + glob.glob('{}/*.{}'.format(self.image_path, file_ext))

        with tqdm(enumerate(sorted(wsipaths))) as t:

            all_image_tiles_hr = []
            all_image_tiles_lr1 = []
            all_image_tiles_lr2 = []

            for wj, wsipath in t:
                t.set_description('Loading wsis.. {:d}/{:d}'.format(1 + wj, len(wsipaths)))

                'generate tiles for this wsi'
                image_tiles_hr, image_tiles_lr1, image_tiles_lr2 = self.get_wsi_patches(wsipath)

                # Check if patches are generated or not for a wsi
                if len(image_tiles_hr) == len(image_tiles_lr1) == len(image_tiles_lr2) == 0:
                    print('bad wsi, no patches are generated for', str(wsipath))
                    continue
                else:
                    all_image_tiles_hr.append(image_tiles_hr)
                    all_image_tiles_lr1.append(image_tiles_lr1)
                    all_image_tiles_lr2.append(image_tiles_lr2)

            # Stack all patches across images
            all_image_tiles_hr = np.concatenate(all_image_tiles_hr)
            all_image_tiles_lr1 = np.concatenate(all_image_tiles_lr1)
            all_image_tiles_lr2 = np.concatenate(all_image_tiles_lr2)

        # # Store WSI patches
        # np.save(os.path.join(self.output_path, 'patches' + str(0)), all_image_tiles_hr)
        # np.save(os.path.join(self.output_path, 'patches' + str(1)), all_image_tiles_lr1)
        # np.save(os.path.join(self.output_path, 'patches' + str(2)), all_image_tiles_lr2)

        return all_image_tiles_hr, all_image_tiles_lr1, all_image_tiles_lr2

    def __getitem__(self, scan, x, y, filename, patch_id):

        'read low res. image'
        m = scan.level_downsamples[self.lr_level_2]

        'save images'
        savepth = '{}/{}/{}'.format(self.output_path, filename, patch_id)
        os.makedirs(savepth, exist_ok=True)
        os.makedirs('{}/hr/'.format(savepth), exist_ok=True)
        os.makedirs('{}/lr1/'.format(savepth), exist_ok=True)
        os.makedirs('{}/lr2/'.format(savepth), exist_ok=True)

        'Extract multi-resolution concentric patches'

        'lr patch_2'
        image_tile_lr2 = scan.read_region((int(m * x), int(m * y)), self.lr_level_2, (self.tile_w, self.tile_h)).convert('RGB')
        image_tile_lr2 = np.array(image_tile_lr2).astype('uint8')

        # Save for visualization
        scan.read_region((int(m * x), int(m * y)), self.lr_level_2, (self.tile_w, self.tile_h)).convert('RGB').save('{}/lr2/{}.png'.format(savepth, patch_id))

        'lr patch_1'
        mlr = scan.level_downsamples[self.lr_level_1]
        left, up = int(int(int(int(m * (x + (self.tile_w / 2))) / mlr) - int(self.tile_w / 2)) * mlr), int(int(int(int(m * (y + (self.tile_h / 2))) / mlr) - int(self.tile_h / 2)) * mlr)

        image_tile_lr1 = scan.read_region((left, up), self.lr_level_1, (self.tile_w, self.tile_h)).convert('RGB')
        image_tile_lr1 = np.array(image_tile_lr1).astype('uint8')

        # Save for visualization
        scan.read_region(
            (left, up),
            self.lr_level_1,
            (self.tile_w, self.tile_h)).convert('RGB').save('{}/lr1/{}.png'.format(savepth, patch_id))

        'hr patch'
        mhr = scan.level_downsamples[self.hr_level]
        left_hr, up_hr = int(int(int(int(m * (x + (self.tile_w / 2))) / mhr) - int(self.tile_w / 2)) * mhr), int(int(int(int(m * (y + (self.tile_h / 2))) / mhr) - int(self.tile_h / 2)) * mhr)

        image_tile_hr = scan.read_region((left_hr, up_hr), self.hr_level, (self.tile_w, self.tile_h)).convert('RGB')
        image_tile_hr = np.array(image_tile_hr).astype('uint8')

        # Save for visualization
        scan.read_region(
            (left_hr, up_hr),
            self.hr_level,
            (self.tile_w, self.tile_h)).convert('RGB').save('{}/hr/{}.png'.format(savepth, patch_id))

        return image_tile_lr2, image_tile_lr1, image_tile_hr

    def get_wsi_patches(self, wsipth):

        'read the wsi scan'
        filename = os.path.basename(wsipth)
        scan = openslide.OpenSlide(wsipth)

        'if a slide has less levels than our desired scan level, ignore the slide'
        if scan.level_count >= 3:

            'downsample multiplier'
            '''
            due to the way pyramid images are stored,
            it's best to use the lower resolution to
            specify the coordinates then pick high res.
            from that (because low. res. pts will always
            be on high res image but when high res coords
            are downsampled, you might lose that (x,y) point)
            '''

            iw, ih = scan.level_dimensions[self.lr_level_2]
            sh, sw = self.tile_stride_h, self.tile_stride_w
            ph, pw = self.tile_h, self.tile_w

            patch_id = 0
            image_tiles_hr = []
            image_tiles_lr1 = []
            image_tiles_lr2 = []

            for ypos in range(sh, ih - 1 - ph, sh):
                for xpos in range(sw, iw - 1 - pw, sw):
                    xph, yph = int(scan.level_downsamples[self.lr_level_2] * xpos), int(scan.level_downsamples[self.lr_level_2] * ypos)
                    if isforeground(self, scan, xph, yph, self.lr_level_2):  # Select valid foreground patch
                        image_tile_lr2, image_tile_lr1, image_tile_hr = self.__getitem__(scan, xpos, ypos, filename, patch_id)

                        image_tiles_hr.append(image_tile_hr)
                        image_tiles_lr1.append(image_tile_lr1)
                        image_tiles_lr2.append(image_tile_lr2)

                        patch_id = patch_id + 1

            # Concatenate
            if len(image_tiles_hr) == len(image_tiles_lr1) == len(image_tiles_lr2) == 0:
                image_tiles_hr == image_tiles_lr1 == image_tiles_lr2 == []
            else:
                image_tiles_hr = np.stack(image_tiles_hr, axis=0).astype('uint8')
                image_tiles_lr1 = np.stack(image_tiles_lr1, axis=0).astype('uint8')
                image_tiles_lr2 = np.stack(image_tiles_lr2, axis=0).astype('uint8')

        return image_tiles_hr, image_tiles_lr1, image_tiles_lr2


####################################################################################################################

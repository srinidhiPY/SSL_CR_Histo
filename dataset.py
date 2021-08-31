import numpy as np
import torch
import os
import openslide
import glob
import random
import h5py
from os.path import exists
from skimage import color
import copy
from skimage.color import rgb2hed
from skimage.color import hed2rgb
from tqdm import tqdm
from util import isforeground, Annotation
from torch.utils.data import Dataset, TensorDataset
from models.randaugment import RandAugment
from torchvision import transforms
from PIL import Image
from albumentations import Compose, Rotate, CenterCrop, HorizontalFlip, RandomScale, Flip, Resize, ShiftScaleRotate, \
    RandomCrop, IAAAdditiveGaussianNoise, ElasticTransform, HueSaturationValue, LongestMaxSize, RandomBrightnessContrast, Blur


#####################################################

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


###############################  List of data augmentation for Pre-training #####################################

def colour_augmentation(image, h_mean, h_std, d_mean, d_std, e_mean, e_std):

    ihc_hed = rgb2hed(image)
    Im_size = image.shape[1]

    h = ihc_hed[:, :, 0]
    d = ihc_hed[:, :, 1]
    e = ihc_hed[:, :, 2]

    hFlat = np.ravel(h, order='A')
    dFlat = np.ravel(d, order='A')
    eFlat = np.ravel(e, order='A')

    # Method
    hmod = random.normalvariate(h_mean, h_std)
    dmod = random.normalvariate(d_mean, d_std)
    emod = random.normalvariate(e_mean, e_std)

    for x in range(len(h.ravel())):
        hFlat[x] = hFlat[x] + hmod
        dFlat[x] = dFlat[x] + dmod
        eFlat[x] = eFlat[x] + emod

    ##############
    h = hFlat.reshape(Im_size, Im_size)
    d = dFlat.reshape(Im_size, Im_size)
    e = eFlat.reshape(Im_size, Im_size)

    zdh = np.stack((h, d, e), 2)
    zdh = hed2rgb(zdh)
    zdh_8bit = (zdh * 255).astype('uint8')
    image = zdh_8bit
    return image

def HSV(img):
    transform = Compose([HueSaturationValue(hue_shift_limit=(-0.1, 0.1), sat_shift_limit=(-1, 1))])
    Aug_img = transform(image=img)
    return Aug_img

def Noise(img):
    transform = Compose([IAAAdditiveGaussianNoise(loc=0, scale=(0 * 255, 0.1 * 255))])
    Aug_img = transform(image=img)
    return Aug_img

def Scale_Resize_Crop(img):
    transform = Compose([Rotate(limit=(-90, 90), interpolation=2), RandomScale(scale_limit=(0.8, 1.2), interpolation=2), Resize(img.shape[1] + 20, img.shape[1] + 20, interpolation=2),
                         RandomCrop(img.shape[1], img.shape[1])])
    Aug_img = transform(image=img)
    return Aug_img

def Shift_Scale_Rotate(img):
    transform = Compose([HorizontalFlip(), ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.5, rotate_limit=45, interpolation=2),
                         RandomCrop(img.shape[1], img.shape[1])])
    Aug_img = transform(image=img)
    return Aug_img

def Color(img):
    Aug_img = colour_augmentation(img, h_mean=0, h_std=random.uniform(-0.035, 0.035), d_mean=0, d_std=random.uniform(-0.035, 0.035), e_mean=0, e_std=random.uniform(-0.035, 0.035))
    return Aug_img

def Blur_img(img):
    transform = Compose([Blur(blur_limit=(3, 7))])
    Aug_img = transform(image=img)
    return Aug_img

def Brightness_Contrast(img):
    transform = Compose([RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2))])
    Aug_img = transform(image=img)
    return Aug_img

def Rotate_Crop(img):
    transform = Compose([Rotate(limit=(-90, 90), interpolation=2), CenterCrop(img.shape[1], img.shape[1])])
    Aug_img = transform(image=img)
    return Aug_img

def augment_pool():
    augs = [HSV, Noise, Scale_Resize_Crop, Shift_Scale_Rotate, Color, Blur_img, Brightness_Contrast, Rotate_Crop]
    return augs


#################

class TensorDataset_Transform(Dataset):

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)

        self.tensors = tensors
        self.transform = transform
        self.augment_pool = augment_pool()

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

            #####
            rad_aug_idx = torch.randperm(8)   # randomize total 8 augmentations
            ops = [self.augment_pool[i] for i in rad_aug_idx]
            for op in ops:
                Data_1 = op(Data_1)
                Data_2 = op(Data_2)
                Data_3 = op(Data_3)
                if isinstance(Data_1, dict):
                    Data_1 = Image.fromarray(Data_1['image'])
                    Data_2 = Image.fromarray(Data_2['image'])
                    Data_3 = Image.fromarray(Data_3['image'])
                    Data_1 = np.array(Data_1)
                    Data_2 = np.array(Data_2)
                    Data_3 = np.array(Data_3)
                else:
                    Data_1 = Data_1
                    Data_2 = Data_2
                    Data_3 = Data_3

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

def DatasetWSIs(image_pth, output_pth, tile_h, tile_w, tile_stride_h, tile_stride_w, lwst_level_idx):

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
    albumentations_transform = Compose([])

    # Tensor dataset with transforms
    dataset = TensorDataset_Transform(tensors=(Data_1, Data_2, Data_3, Class_labels),
                                      transform=albumentations_transform)

    return dataset


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

    def __getitem__(self, scan, x, y, filename, patch_id, bias):

        'read low res. image'
        m = scan.level_downsamples[bias + self.lr_level_2]

        'save images'
        savepth = '{}/{}/{}'.format(self.output_path, filename, patch_id)
        os.makedirs(savepth, exist_ok=True)
        os.makedirs('{}/hr/'.format(savepth), exist_ok=True)
        os.makedirs('{}/lr1/'.format(savepth), exist_ok=True)
        os.makedirs('{}/lr2/'.format(savepth), exist_ok=True)

        'lr patch_2'
        image_tile_lr2 = scan.read_region(
            (int(m * x), int(m * y)),
            bias + self.lr_level_2,
            (self.tile_w, self.tile_h)).convert('RGB')

        image_tile_lr2 = np.array(image_tile_lr2).astype('uint8')

        # Save for visualization
        scan.read_region(
            (int(m * x), int(m * y)),
            bias + self.lr_level_2,
            (self.tile_w, self.tile_h)).convert('RGB').save('{}/lr2/{}.png'.format(savepth, patch_id))

        'lr patch_1'
        mlr = scan.level_downsamples[bias + self.lr_level_1]
        left, up = int(int(int(m * (x + (self.tile_w / 2))) / mlr) * mlr), int(
            int(int(m * (y + (self.tile_h / 2))) / mlr) * mlr)

        image_tile_lr1 = scan.read_region(
            (left, up),
            bias + self.lr_level_1,
            (self.tile_w, self.tile_h)).convert('RGB')

        image_tile_lr1 = np.array(image_tile_lr1).astype('uint8')

        # Save for visualization
        scan.read_region(
            (left, up),
            bias + self.lr_level_1,
            (self.tile_w, self.tile_h)).convert('RGB').save('{}/lr1/{}.png'.format(savepth, patch_id))

        'hr patch'
        mhr = scan.level_downsamples[bias + self.hr_level]
        left_hr, up_hr = int(int(int(m * (x + (self.tile_w / 2))) / mhr) * mhr), int(
            int(int(m * (y + (self.tile_h / 2))) / mhr) * mhr)

        image_tile_hr = scan.read_region(
            (left_hr, up_hr),
            bias + self.hr_level,
            (self.tile_w, self.tile_h)).convert('RGB')

        image_tile_hr = np.array(image_tile_hr).astype('uint8')

        # Save for visualization
        scan.read_region(
            (left_hr, up_hr),
            bias + self.hr_level,
            (self.tile_w, self.tile_h)).convert('RGB').save('{}/hr/{}.png'.format(savepth, patch_id))

        return image_tile_lr2, image_tile_lr1, image_tile_hr

    def get_wsi_patches(self, wsipth):

        'read the wsi scan'
        filename = os.path.basename(wsipth)
        scan = openslide.OpenSlide(wsipth)

        'ratio of pixel scale, we work with ~0.5um'
        pixel_scale = np.uint8(np.round(0.5 / float(scan.properties['openslide.mpp-x'])))
        bias = 0  # np.uint8(np.log2(pixel_scale))

        'if a slide has less levels than our desired scan level, ignore the slide'
        if scan.level_count >= 3 and pixel_scale >= 1:

            'gt mask'
            wsi = scan.read_region((0, 0), scan.level_count - self.lwst_level_idx, scan.level_dimensions[-self.lwst_level_idx]).convert('RGB')  # -1 for BreastPathQ, -5 for Camelyon16 (lowest level);
            wsi = np.array(wsi)
            lab = color.rgb2lab(np.asarray(wsi))
            mu = np.mean(lab[..., 1])

            'downsample multiplier'
            '''
            due to the way pyramid images are stored,
            it's best to use the lower resolution to
            specify the coordinates then pick high res.
            from that (because low. res. pts will always
            be on high res image but when high res coords
            are downsampled, you might lose that (x,y) point)
            '''

            iw, ih = scan.level_dimensions[bias + self.lr_level_2]
            sh, sw = self.tile_stride_h, self.tile_stride_w
            ph, pw = self.tile_h, self.tile_w

            patch_id = 0
            image_tiles_hr = []
            image_tiles_lr1 = []
            image_tiles_lr2 = []

            for ypos in range(sh, ih - 1 - ph, sh):
                for xpos in range(sw, iw - 1 - pw, sw):
                    xph, yph = int(scan.level_downsamples[bias + self.lr_level_2] * xpos), int(
                        scan.level_downsamples[bias + self.lr_level_2] * ypos)
                    if isforeground(self, scan, xph, yph, bias + self.lr_level_2, mu):  # Select valid foreground patch
                        image_tile_lr2, image_tile_lr1, image_tile_hr = self.__getitem__(scan, xpos, ypos, filename,
                                                                                         patch_id, bias)

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

#### SSL/SSL_CR - BreastPathQ dataloaders ########

class DatasetBreastPathQ_Supervised_train:

    def __init__(self, dataset_path, image_size, transform=None):

        """
        BreastPathQ dataset: supervised fine-tuning on downstream task
        """

        self.image_size = image_size
        self.transform = transform

        # Resize images
        self.transform1 = Compose([Resize(image_size, image_size, interpolation=2)])  # 256

        # Data augmentations
        self.transform4 = Compose([Rotate(limit=(-90, 90), interpolation=2), CenterCrop(image_size, image_size)])
        self.transform5 = Compose(
            [Rotate(limit=(-90, 90), interpolation=2), RandomScale(scale_limit=(0.8, 1.2), interpolation=2),
             Resize(image_size + 20, image_size + 20, interpolation=2),
             RandomCrop(image_size, image_size)])

        self.datalist = []
        data_paths = glob.glob(dataset_path + "*.h5")
        with tqdm(enumerate(sorted(data_paths)), disable=True) as t:
            for wj, data_path in t:
                data = h5py.File(data_path)
                data_patches = data['x'][:]
                cls_id = data['y'][:]
                for idx in range(len(data_patches)):
                    self.datalist.append((data_patches[idx], cls_id[idx]))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        np_data = self.datalist[index][0]
        np_data = np.transpose(np_data, (1, 2, 0))
        img = Image.fromarray((np_data * 255).astype(np.uint8))

        # label assignment
        label = self.datalist[index][1]

        if self.transform:

            # Convert PIL image to numpy array
            img = np.array(img)

            # First image
            img1 = self.transform1(image=img)
            img1 = Image.fromarray(img1['image'])
            img1 = np.array(img1)

            Aug1_img1 = self.transform4(image=img1)
            Aug2_img1 = self.transform5(image=img1)

            # Convert numpy array to PIL Image
            img1 = Image.fromarray(img1)
            Aug1_img1 = Image.fromarray(Aug1_img1['image'])
            Aug2_img1 = Image.fromarray(Aug2_img1['image'])

            # Convert to numpy array
            img1 = np.array(img1)
            Aug1_img1 = np.array(Aug1_img1)
            Aug2_img1 = np.array(Aug2_img1)

            # Stack along specified dimension
            img = np.stack((img1, Aug1_img1, Aug2_img1), axis=0)

            # Numpy to torch
            img = torch.from_numpy(img)

            # Randomize the augmentations
            shuffle_idx = torch.randperm(len(img))
            img = img[shuffle_idx, :, :, :]

            label = np.array(label)
            label = torch.from_numpy(label)
            label = label.repeat(img.shape[0])

            # Change Tensor Dimension to N x C x H x W
            img = img.permute(0, 3, 1, 2)

        return img, label

#########
class DatasetBreastPathQ_eval:

    def __init__(self, dataset_path, image_size, transform=None):

        """
        BreastPathQ dataset: test
        """

        self.image_size = image_size
        self.transform = transform

        dataset_pathA = os.path.join(dataset_path, "TestSetSherine/")
        dataset_pathB = os.path.join(dataset_path, "TestSetSharon/")

        self.datalist = []
        data_pathsA = glob.glob(dataset_pathA + "*.h5")
        data_pathsB = glob.glob(dataset_pathB + "*.h5")

        with tqdm(enumerate(sorted(zip(data_pathsA, data_pathsB))), disable=True) as t:
            for wj, (data_pathA, data_pathB) in t:

                dataA = h5py.File(data_pathA)
                dataB = h5py.File(data_pathB)

                data_patches = dataA['x'][:]

                cls_idA = dataA['y'][:]
                cls_idB = dataB['y'][:]

                for idx in range(len(data_patches)):
                    self.datalist.append((data_patches[idx], cls_idA[idx], cls_idB[idx]))

    def __len__(self):

        return len(self.datalist)

    def __getitem__(self, index):

        np_data = self.datalist[index][0]
        np_data = np.transpose(np_data, (1, 2, 0))
        img = Image.fromarray((np_data * 255).astype(np.uint8))

        # label assignment
        labelA = self.datalist[index][1]
        labelB = self.datalist[index][2]

        if self.transform:
            img = self.transform(img)
            img = np.array(img)
            img = torch.from_numpy(img)

            labelA = np.array(labelA)
            labelA = torch.from_numpy(labelA)

            labelB = np.array(labelB)
            labelB = torch.from_numpy(labelB)

            # Change Tensor Dimension to N x C x H x W
            img = img.permute(2, 0, 1)

        return img, labelA, labelB

##################

class DatasetBreastPathQ_SSLtrain(Dataset):

    """ BreastPathQ consistency training / validation  """

    def __init__(self, dataset_path, transform=None):

        self.datalist = []
        self.transform = transform

        data_paths = glob.glob(dataset_path + "*.h5")
        with tqdm(enumerate(sorted(data_paths)), disable=True) as t:
            for wj, data_path in t:
                data = h5py.File(data_path)
                data_patches = data['x'][:]
                cls_id = data['y'][:]
                for idx in range(len(data_patches)):
                    self.datalist.append((data_patches[idx], cls_id[idx]))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        np_data = self.datalist[index][0]
        np_data = np.transpose(np_data, (1, 2, 0))
        img = Image.fromarray((np_data * 255).astype(np.uint8))
        label = self.datalist[index][1]

        if self.transform:
            image = self.transform(img)

            if isinstance(image, tuple):
                img = image[0]
                target = image[1]

                # Numpy to torch
                img = np.array(img)
                img = torch.from_numpy(img)
                target = np.array(target)
                target = torch.from_numpy(target)

                # Change Tensor Dimension to N x C x H x W
                img = img.permute(2, 0, 1)
                target = target.permute(2, 0, 1)

            else:
                # Numpy to torch
                img = np.array(image)
                img = torch.from_numpy(img)

                target = np.array(label)
                target = torch.from_numpy(target)

                # Change Tensor Dimension to N x C x H x W
                img = img.permute(2, 0, 1)

        return img, target

#######################

class TransformFix(object):

    """" Weak and strong augmentation for consistency training """

    def __init__(self, image_size, N):
        self.weak = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(size=image_size)])
        self.strong = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(size=image_size),
                                          RandAugment(n=N, m=10)])  # (7) - standard value for BreastPathQ/Camelyon16/Kather

    def __call__(self, x):
        weak_img = self.weak(x)
        # weak_img.show()
        strong_img = self.strong(x)
        # strong_img.show()
        return weak_img, strong_img

#######################


#############################################################################
'Camelyon16 dataset dataloaders (train, val and test)'

class DatasetCamelyon16_Supervised_train(Dataset):

    def __init__(self, data_path, json_path, image_size, transform=None):

            """ Camelyon16 dataset: supervised fine-tuning on downstream task """

            self.transform = transform
            self.data_path = data_path
            self.json_path = json_path
            self._preprocess()

            # Data augmentations
            self.transform1 = Compose([Rotate(limit=(-90, 90), interpolation=2), CenterCrop(image_size, image_size)])
            self.transform2 = Compose([Rotate(limit=(-90, 90), interpolation=2), RandomScale(scale_limit=(0.8, 1.2), interpolation=2),
                                       Resize(image_size + 20, image_size + 20, interpolation=2), RandomCrop(image_size, image_size)])

    def _preprocess(self):

        self.pids = list(map(lambda x: x.strip('.json'), os.listdir(self.json_path)))

        self.annotations = {}
        for pid in self.pids:
            pid_json_path = os.path.join(self.json_path, pid + '.json')
            anno = Annotation()
            anno.from_json(pid_json_path)
            self.annotations[pid] = anno

        self.coords = []
        f = open(os.path.join(self.data_path, 'list.txt'))

        for line in f:
            pid, x_center, y_center = line.strip('\n').split(',')[0:3]
            # Split into fine-tune set 210 WSIs - (Normal set - 36 to 160; Tumor set - 26 to 110)  Note: rest is used as pre-training
            name_split = pid.split("_")
            if name_split[0] == 'Tumor' and int(name_split[1]) > 25:
                x_center, y_center = int(x_center), int(y_center)
                self.coords.append((pid, x_center, y_center))
            elif name_split[0] == 'Normal' and int(name_split[1]) > 35:
                x_center, y_center = int(x_center), int(y_center)
                self.coords.append((pid, x_center, y_center))
            else:
                continue
        f.close()

        self.num_image = len(self.coords)

    def __len__(self):
        return self.num_image

    def __getitem__(self, idx):

        pid, x_center, y_center = self.coords[idx]
        img = Image.open(os.path.join(self.data_path, '{}.png'.format(idx)))

        # Assign Label as tumor or normal
        if self.annotations[pid].inside_polygons((x_center, y_center), True):
            label = 1
        else:
            label = 0

        if self.transform:

            # Convert PIL image to numpy array
            img = np.array(img)

            Aug1_img1 = self.transform1(image=img)
            Aug2_img1 = self.transform2(image=img)

            # Convert numpy array to PIL Image
            img1 = Image.fromarray(img)
            Aug1_img1 = Image.fromarray(Aug1_img1['image'])
            Aug2_img1 = Image.fromarray(Aug2_img1['image'])

            # Convert to numpy array
            img1 = np.array(img1)
            Aug1_img1 = np.array(Aug1_img1)
            Aug2_img1 = np.array(Aug2_img1)

            # Stack along specified dimension
            img1 = np.stack((img1, Aug1_img1, Aug2_img1), axis=0)

            # Numpy to torch
            img1 = torch.from_numpy(img1)

            # Randomize the augmentations
            shuffle_idx = torch.randperm(len(img1))
            img1 = img1[shuffle_idx, :, :, :]

            label = np.array(label)
            label = torch.from_numpy(label)
            label = label.repeat(img1.shape[0])

            # Change Tensor Dimension to N x C x H x W
            img = img1.permute(0, 3, 1, 2)

        return img, label

###########

class DatasetCamelyon16_SSLtrain(Dataset):

    def __init__(self, data_path, json_path, transform=None):

            """ Camelyon16 dataloaders for consistency training """

            self.transform = transform
            self.data_path = data_path
            self.json_path = json_path
            self._preprocess()

    def _preprocess(self):

        self.pids = list(map(lambda x: x.strip('.json'), os.listdir(self.json_path)))

        self.annotations = {}
        for pid in self.pids:
            pid_json_path = os.path.join(self.json_path, pid + '.json')
            anno = Annotation()
            anno.from_json(pid_json_path)
            self.annotations[pid] = anno

        self.coords = []
        f = open(os.path.join(self.data_path, 'list.txt'))

        for line in f:
            pid, x_center, y_center = line.strip('\n').split(',')[0:3]
            # Split into fine-tune set 210 WSIs - (Normal set - 36 to 160; Tumor set - 26 to 110)
            name_split = pid.split("_")
            if name_split[0] == 'Tumor' and int(name_split[1]) > 25:
                x_center, y_center = int(x_center), int(y_center)
                self.coords.append((pid, x_center, y_center))
            elif name_split[0] == 'Normal' and int(name_split[1]) > 35:
                x_center, y_center = int(x_center), int(y_center)
                self.coords.append((pid, x_center, y_center))
            else:
                continue
        f.close()

        self.num_image = len(self.coords)

    def __len__(self):
        return self.num_image

    def __getitem__(self, idx):

        pid, x_center, y_center = self.coords[idx]
        img = Image.open(os.path.join(self.data_path, '{}.png'.format(idx)))

        # Assign label as tumor or normal
        if self.annotations[pid].inside_polygons((x_center, y_center), True):
            label = 1
        else:
            label = 0

        if self.transform:
            image = self.transform(img)

            if isinstance(image, tuple):
                img = image[0]
                target = image[1]

                # Numpy to torch
                img = np.array(img)
                img = torch.from_numpy(img)
                target = np.array(target)
                target = torch.from_numpy(target)

                # Change Tensor Dimension to N x C x H x W
                img = img.permute(2, 0, 1)
                target = target.permute(2, 0, 1)

            else:
                # Numpy to torch
                img = np.array(image)
                img = torch.from_numpy(img)

                target = np.array(label)
                target = torch.from_numpy(target)

                # Change Tensor Dimension to N x C x H x W
                img = img.permute(2, 0, 1)

        return img, target

####################

class DatasetCamelyon16_eval(Dataset):

    def __init__(self, data_path, json_path):

            """
            Camelyon16 dataset for validation
            """

            self.data_path = data_path
            self.json_path = json_path
            self._preprocess()

    def _preprocess(self):

        self.pids = list(map(lambda x: x.strip('.json'), os.listdir(self.json_path)))

        self.annotations = {}
        for pid in self.pids:
            pid_json_path = os.path.join(self.json_path, pid + '.json')
            anno = Annotation()
            anno.from_json(pid_json_path)
            self.annotations[pid] = anno

        self.coords = []
        f = open(os.path.join(self.data_path, 'list.txt'))

        for line in f:
            pid, x_center, y_center = line.strip('\n').split(',')[0:3]
            # Split into fine-tune set 210 WSIs - (Normal set - 36 to 160; Tumor set - 26 to 110)
            name_split = pid.split("_")
            if name_split[0] == 'Tumor' and int(name_split[1]) > 25:
                x_center, y_center = int(x_center), int(y_center)
                self.coords.append((pid, x_center, y_center))
            elif name_split[0] == 'Normal' and int(name_split[1]) > 35:
                x_center, y_center = int(x_center), int(y_center)
                self.coords.append((pid, x_center, y_center))
            else:
                continue
        f.close()

        self.num_image = len(self.coords)

    def __len__(self):
        return self.num_image

    def __getitem__(self, idx):

        pid, x_center, y_center = self.coords[idx]
        img = Image.open(os.path.join(self.data_path, '{}.png'.format(idx)))

        # Label
        if self.annotations[pid].inside_polygons((x_center, y_center), True):
            label = 1
        else:
            label = 0

        # Convert PIL image to numpy array
        img = np.array(img)

        # Numpy to torch
        img = torch.from_numpy(img)

        label = np.array(label)
        label = torch.from_numpy(label)

        # Change Tensor Dimension to N x C x H x W
        img = img.permute(2, 0, 1)

        return img, label

#################

class DatasetCamelyon16_test(Dataset):                           # Final Predictions #

    def __init__(self, data_path, mask_path, image_size):

            """
            Camelyon16 dataset class wrapper
                    data_path: string, path to pre-sampled images
                    json_path: string, path to the annotations in json format
            """

            self.data_path = data_path
            self.mask_path = mask_path
            self.image_size = image_size
            self.preprocess()

    def preprocess(self):

        self.mask = np.load(self.mask_path)
        self.slide = openslide.OpenSlide(self.data_path)

        X_slide, Y_slide = self.slide.level_dimensions[0]
        X_mask, Y_mask = self.mask.shape

        if round(X_slide / X_mask) != round(Y_slide / Y_mask):
            raise Exception('Slide/Mask dimension does not match ,'
                            ' X_slide / X_mask : {} / {},'
                            ' Y_slide / Y_mask : {} / {}'
                            .format(X_slide, X_mask, Y_slide, Y_mask))

        self.resolution = round(X_slide * 1.0 / X_mask)
        if not np.log2(self.resolution).is_integer():
            raise Exception('Resolution (X_slide / X_mask) is not power of 2 :'
                            ' {}'.format(self.resolution))

        # all the indices for tissue region from the tissue mask
        self.X_idcs, self.Y_idcs = np.where(self.mask)

    def __len__(self):
        return len(self.X_idcs)

    def __getitem__(self, idx):

        x_mask, y_mask = self.X_idcs[idx], self.Y_idcs[idx]

        x_center = int((x_mask) * self.resolution)
        y_center = int((y_mask) * self.resolution)

        x = int(x_center - self.image_size / 2)
        y = int(y_center - self.image_size / 2)

        img = self.slide.read_region((x, y), 0, (self.image_size, self.image_size)).convert('RGB')
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        return img, x_mask, y_mask

##########################################################################

###### Kather dataloaders #############

class DatasetKather_eval:

    def __init__(self, dataset_path, image_size):

        """
        Kather dataset class wrapper (test/val)
        """

        self.image_size = image_size

        # Resize images
        self.transform1 = Compose([Resize(image_size, image_size, interpolation=2)])

        self.datalist = []
        cls_paths = glob.glob('{}/*/'.format(dataset_path))
        with tqdm(enumerate(sorted(cls_paths)), disable=True) as t:
            for wj, cls_path in t:
                cls_id = str(os.path.split(os.path.dirname(cls_path))[-1])
                patch_pths = glob.glob('{}/*.tif'.format(cls_path))
                for pth in patch_pths:
                    self.datalist.append((pth, cls_id))

    def __len__(self):

        return len(self.datalist)

    def __getitem__(self, index):

        img = Image.open(self.datalist[index][0])

        # label assignment
        label = str(self.datalist[index][1])

        if label == 'ADI':
            label = 0
        elif label == 'BACK':
            label = 1
        elif label == 'DEB':
            label = 2
        elif label == 'LYM':
            label = 3
        elif label == 'MUC':
            label = 4
        elif label == 'MUS':
            label = 5
        elif label == 'NORM':
            label = 6
        elif label == 'STR':
            label = 7
        else:
            label = 8  # 'TUM'

        # Convert PIL image to numpy array
        img = np.array(img)

        img = self.transform1(image=img)

        # Convert numpy array to PIL Image
        img = Image.fromarray(img['image'])

        img = np.array(img)
        img = torch.from_numpy(img)

        label = np.array(label)
        label = torch.from_numpy(label)

        # Change Tensor Dimension to N x C x H x W
        img = img.permute(2, 0, 1)

        return img, label

#############

class DatasetKather_Supervised_train:

    def __init__(self, dataset_path, image_size):

        """
        Kather dataset: supervised fine-tuning on downstream task
        """

        self.image_size = image_size

        # Resize images
        self.transform1 = Compose([Resize(image_size, image_size, interpolation=2)])

        # Data augmentations
        self.transform4 = Compose([Rotate(limit=(-90, 90), interpolation=2), CenterCrop(image_size, image_size)])
        self.transform5 = Compose([Rotate(limit=(-90, 90), interpolation=2), RandomScale(scale_limit=(0.8, 1.2), interpolation=2),
                                   Resize(image_size + 20, image_size + 20, interpolation=2), RandomCrop(image_size, image_size)])

        self.datalist = []
        cls_paths = glob.glob('{}/*/'.format(dataset_path))
        with tqdm(enumerate(sorted(cls_paths)), disable=True) as t:
            for wj, cls_path in t:
                cls_id = str(os.path.split(os.path.dirname(cls_path))[-1])
                patch_pths = glob.glob('{}/*.tif'.format(cls_path))
                for pth in patch_pths:
                    self.datalist.append((pth, cls_id))

    def __len__(self):

        return len(self.datalist)

    def __getitem__(self, index):

        img = Image.open(self.datalist[index][0])

        # label assignment
        label = str(self.datalist[index][1])

        if label == 'ADI':
            label = 0
        elif label == 'BACK':
            label = 1
        elif label == 'DEB':
            label = 2
        elif label == 'LYM':
            label = 3
        elif label == 'MUC':
            label = 4
        elif label == 'MUS':
            label = 5
        elif label == 'NORM':
            label = 6
        elif label == 'STR':
            label = 7
        else:
            label = 8  # 'TUM'

        #################
        # Convert PIL image to numpy array
        img = np.array(img)

        # First image
        img1 = self.transform1(image=img)
        img1 = Image.fromarray(img1['image'])
        img1 = np.array(img1)

        Aug1_img1 = self.transform4(image=img1)
        Aug2_img1 = self.transform5(image=img1)

        # Convert numpy array to PIL Image
        img1 = Image.fromarray(img1)
        # img1.show()
        Aug1_img1 = Image.fromarray(Aug1_img1['image'])
        # Aug1_img1.show()
        Aug2_img1 = Image.fromarray(Aug2_img1['image'])
        # Aug2_img1.show()

        # Convert to numpy array
        img1 = np.array(img1)
        Aug1_img1 = np.array(Aug1_img1)
        Aug2_img1 = np.array(Aug2_img1)

        # Stack along specified dimension
        img = np.stack((img1, Aug1_img1, Aug2_img1), axis=0)

        # Numpy to torch
        img = torch.from_numpy(img)

        # Randomize the augmentations
        shuffle_idx = torch.randperm(len(img))
        img = img[shuffle_idx, :, :, :]

        label = np.array(label)
        label = torch.from_numpy(label)
        label = label.repeat(img1.shape[0])

        # Change Tensor Dimension to N x C x H x W
        img = img.permute(0, 3, 1, 2)

        return img, label

#############

class DatasetKather_SSLtrain:

    def __init__(self, dataset_path, image_size, transform=None):

        """
        Kather dataset: consistency training on downstream task
        """

        self.image_size = image_size
        self.transform = transform

        # Resize images
        self.transform1 = Compose([Resize(image_size, image_size, interpolation=2)])

        self.datalist = []
        cls_paths = glob.glob('{}/*/'.format(dataset_path))
        with tqdm(enumerate(sorted(cls_paths)), disable=True) as t:
            for wj, cls_path in t:
                cls_id = str(os.path.split(os.path.dirname(cls_path))[-1])
                patch_pths = glob.glob('{}/*.tif'.format(cls_path))
                for pth in patch_pths:
                    self.datalist.append((pth, cls_id))

    def __len__(self):

        return len(self.datalist)

    def __getitem__(self, index):

        img = Image.open(self.datalist[index][0])

        if self.transform:
            img = np.array(img)
            img = self.transform1(image=img)
            img = Image.fromarray(img['image'])
            image = self.transform(img)

            if isinstance(image, tuple):
                img = image[0]
                target = image[1]

                # Numpy to torch
                img = np.array(img)
                img = torch.from_numpy(img)
                target = np.array(target)
                target = torch.from_numpy(target)

                # Change Tensor Dimension to N x C x H x W
                img = img.permute(2, 0, 1)
                target = target.permute(2, 0, 1)

            else:
                # Numpy to torch
                img = np.array(image)
                img = torch.from_numpy(img)

                target = np.array(label)
                target = torch.from_numpy(target)

                # Change Tensor Dimension to N x C x H x W
                img = img.permute(2, 0, 1)

        return img, target
#############################

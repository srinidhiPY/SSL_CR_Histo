import random
import numpy as np
import torch
import PIL
import cv2 as cv
from PIL import Image
from skimage import color
import copy
from skimage.color import rgb2hed
from skimage.color import hed2rgb
from albumentations import Compose, Rotate, CenterCrop, HorizontalFlip, RandomScale, Flip, Resize, ShiftScaleRotate, \
    RandomCrop, IAAAdditiveGaussianNoise, ElasticTransform, HueSaturationValue, LongestMaxSize, RandomBrightnessContrast, \
    Blur


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


def HSV(img, v):  # [-1, 1]
    assert -1 <= v <= 1
    if random.random() < 0.5:
        v = -v
    transform = Compose([HueSaturationValue(hue_shift_limit=v, sat_shift_limit=v, val_shift_limit=v)])
    Aug_img = transform(image=img)
    return Aug_img

def Noise(img, v):  # [0, 0.15]
    assert 0 <= v <= 0.15
    transform = Compose([IAAAdditiveGaussianNoise(loc=0, scale=(0 * 255, v * 255))])
    Aug_img = transform(image=img)
    return Aug_img

def Scale_Resize_Crop(img, v):  # [0.8, 1.2]
    assert 0.8 <= v <= 1.2
    transform = Compose([RandomScale(scale_limit=v, interpolation=2), Resize(img.shape[1] + 20, img.shape[1] + 20, interpolation=2),
                         RandomCrop(img.shape[1], img.shape[1])])
    Aug_img = transform(image=img)
    return Aug_img

def Shift_Scale_Rotate(img, v):  # [0.01, 0.1]
    assert 0.01 <= v <= 0.1
    if random.random() < 0.5:
        v = -v
    transform = Compose([ShiftScaleRotate(shift_limit=v, scale_limit=v+0.5, rotate_limit=90, interpolation=2),
                         RandomCrop(img.shape[1], img.shape[1])])
    Aug_img = transform(image=img)
    return Aug_img

def Color(img, v):  # [-0.035, 0.035]
    assert -0.035 <= v <= 0.035
    Aug_img = colour_augmentation(img, h_mean=0, h_std=random.uniform(-0.035, 0.035), d_mean=0, d_std=random.uniform(-0.035, 0.035), e_mean=0, e_std=random.uniform(-0.035, 0.035))
    return Aug_img

def Blur_img(img, v):  # [0, 2]
    assert 0 <= v <= 2
    transform = Compose([Blur(blur_limit=int(v+5))])
    Aug_img = transform(image=img)
    return Aug_img

def Brightness(img, v):  # [-0.2, 0.2]
    assert -0.2 <= v <= 0.2
    transform = Compose([RandomBrightnessContrast(brightness_limit=v)])
    Aug_img = transform(image=img)
    return Aug_img

def Contrast(img, v):  # [-0.2, 0.2]
    assert -0.2 <= v <= 0.2
    transform = Compose([RandomBrightnessContrast(contrast_limit=v)])
    Aug_img = transform(image=img)
    return Aug_img

def Rotate_Crop(img, v):  # [-90, 90]
    assert -90 <= v <= 90
    if random.random() < 0.5:
        v = -v
    transform = Compose([Flip(), Rotate(limit=v, interpolation=2), CenterCrop(img.shape[1], img.shape[1])])
    Aug_img = transform(image=img)
    return Aug_img

def augment_pool():
    augs = [(HSV, -1, 1),
            (Noise, 0, 0.15),
            (Scale_Resize_Crop, 0.8, 1.2),
            (Shift_Scale_Rotate, 0.01, 0.1),
            (Color, -0.035, 0.035),
            (Blur_img, 0, 2),
            (Brightness, -0.2, 0.2),
            (Contrast, -0.2, 0.2),
            (Rotate_Crop, -90, 90),
            ]
    return augs

class RandAugment(object):

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.augment_pool = augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, minval, maxval in ops:
            v = np.random.randint(1, self.m)
            val = (float(v) / 30) * float(maxval - minval) + minval
            img = np.array(img)
            img = op(img, val)
            if isinstance(img, dict):
                img = Image.fromarray(img['image'])
                img = np.array(img)
            else:
                img = img
        return Image.fromarray(img)

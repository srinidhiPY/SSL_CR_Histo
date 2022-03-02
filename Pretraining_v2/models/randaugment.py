import random
import numpy as np
import torch
import PIL
from PIL import Image, ImageEnhance, ImageOps
from models.augmenters.color.hsbcoloraugmenter import HsbColorAugmenter
from models.augmenters.color.hedcoloraugmenter import HedColorAugmenter

"""AutoAugment and RandAugment policies for enhanced image preprocessing.
[1] AutoAugment Reference: https://arxiv.org/abs/1805.09501
[2] RandAugment Reference: https://arxiv.org/abs/1909.13719
[3] Tailoring automated data augmentation to H&E-stained histopathology: https://proceedings.mlr.press/v143/faryna21a.html
[4] Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology: https://www.sciencedirect.com/science/article/abs/pii/S1361841519300799

Please cite the following above papers if use this RandAugment code. The RandAugment code and the choice of hyper-parameters are directly adopted from [3-4].
"""

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme (Refer [3]).
MAX_LEVEL = 10


######
def _randomly_negate_tensor(tensor):
    """With 50% prob turn the tensor negative."""
    rand_cva = list([1, 0])

    should_flip = random.choice(rand_cva)

    if should_flip == 1:
        final_tensor = tensor
    else:
        final_tensor = -tensor
    return final_tensor


################
def identity(image, factor):
    """Implements Identity
    """
    return image


def contrast(image, factor):
  factor = (factor / MAX_LEVEL) * 1.8 + 0.1
  """Equivalent of PIL Contrast."""
  image = Image.fromarray(image)
  image = ImageEnhance.Contrast(image).enhance(factor)
  return np.asarray(image)


def brightness(image, factor):
  factor = (factor / MAX_LEVEL) * 1.8 + 0.1
  """Equivalent of PIL Brightness."""
  image = Image.fromarray(image)
  image = ImageEnhance.Brightness(image).enhance(factor)
  return np.asarray(image)


def sharpness(image, factor):
  factor = (factor / MAX_LEVEL) * 1.8 + 0.1
  """Implements Sharpness function from PIL using TF ops."""
  image = Image.fromarray(image)
  image = ImageEnhance.Sharpness(image).enhance(factor)
  return np.asarray(image)


def rotate(image, degrees):
    degrees = (degrees / MAX_LEVEL) * 30.
    degrees = _randomly_negate_tensor(degrees)
    """Equivalent of PIL Posterize."""
    image = Image.fromarray(image)
    image = image.rotate(angle=degrees)
    return np.asarray(image)


def translate_x(image, pixels):
    pixels = (pixels / MAX_LEVEL) * float(10)
    # Flip level to negative with 50% chance.
    pixels = _randomly_negate_tensor(pixels)
    """Equivalent of PIL Translate in X dimension."""
    image = Image.fromarray(image)
    image=image.transform(image.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0))
    return np.asarray(image)


def translate_y(image, pixels):
    pixels = (pixels / MAX_LEVEL) * float(10)
    # Flip level to negative with 50% chance.
    pixels = _randomly_negate_tensor(pixels)
    """Equivalent of PIL Translate in Y dimension."""
    image = Image.fromarray(image)
    image=image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels))
    return np.asarray(image)


def shear_x(image, level):
  level = (level/MAX_LEVEL) * 0.3
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  """Equivalent of PIL Shearing in X dimension."""
  # Shear parallel to x axis is a projective transform
  # with a matrix form of:
  # [1  level
  #  0  1].
  image = Image.fromarray(image)
  image=image.transform(image.size, Image.AFFINE, (1, level, 0, 0, 1, 0), Image.BICUBIC)
  return np.asarray(image)


def shear_y(image, level):
  level = (level/MAX_LEVEL) * 0.3
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  """Equivalent of PIL Shearing in Y dimension."""
  # Shear parallel to y axis is a projective transform
  # with a matrix form of:
  # [1  0
  #  level  1].
  image = Image.fromarray(image)
  image=image.transform(image.size, Image.AFFINE, (1, 0, 0, level,  1, 0), Image.BICUBIC)
  return np.asarray(image)


def hsv(image, factor):
    factor = (factor * 0.03)
    image = np.transpose(image, [2, 0, 1])
    augmentor = HsbColorAugmenter(hue_sigma_range=(-factor, factor), saturation_sigma_range=(-factor, factor),
                                  brightness_sigma_range=(0, 0))
    # To select a random magnitude value between -factor:factor, if commented the m value will be constant
    augmentor.randomize()
    return np.transpose(augmentor.transform(image), [1, 2, 0])


def hed(image, factor):
    factor = (factor * 0.03)
    image = np.transpose(image, [2, 0, 1])
    augmentor = HedColorAugmenter(haematoxylin_sigma_range=(-factor, factor), haematoxylin_bias_range=(-factor, factor),
                                  eosin_sigma_range=(-factor, factor), eosin_bias_range=(-factor, factor),
                                  dab_sigma_range=(-factor, factor), dab_bias_range=(-factor, factor),
                                  cutoff_range=(0.15, 0.85))
    ##To select a random magnitude value between -factor:factor, if commented the m value will be constant
    augmentor.randomize()
    return np.transpose(augmentor.transform(image), [1, 2, 0])


def autocontrast(image, factor):
  """Implements Autocontrast function from PIL using TF ops.
  Args:
    image: A 3D uint8 tensor.
  Returns:
    The image after it has had autocontrast applied to it and will be of type
    uint8.
  """
  image = Image.fromarray(image)
  image = ImageOps.autocontrast(image)
  return np.asarray(image)


def color(image, factor):
  factor = (factor/MAX_LEVEL) * 1.8 + 0.1
  """Equivalent of PIL Color."""
  image = Image.fromarray(image)
  image = ImageEnhance.Color(image).enhance(factor)
  return np.asarray(image)


def equalize(image, factor):
  """Implements Equalize function from PIL using TF ops."""
  image = Image.fromarray(image)
  image = ImageOps.equalize(image)
  return np.asarray(image)


####################
def augment_pool():    # List of histopathology specific augmnentations
    augs = [(identity, 0, 0),
            (contrast, 0.0, 5.5),
            (brightness, 0.0, 5.5),
            (sharpness, 0.0, 5.5),
            (rotate, -90, 90),
            (translate_x, -30, 30),
            (translate_y, -30, 30),
            (shear_x, -0.9, 0.9),
            (shear_y, -0.9, 0.9),
            (hed, -0.9, 0.9),
            (hsv, -0.9, 0.9),
            (autocontrast, 0, 0),
            (color, 0, 5.5),
            (equalize, 0, 0)]
    return augs


#############
class RandAugment(object):

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.count = 0
        self.augment_pool = augment_pool()

    def __call__(self, img):
        ops = random.sample(self.augment_pool, k=self.n)
        for op, minval, maxval in ops:
            val = np.random.uniform(1, self.m)
            print('val', val)
            img = op(img, val)

        # im = Image.fromarray(img)
        # im.save('/home/srinidhi/Research/Code/Tiger_Challenge/SSL/Data/output/Augmentations/' + str(self.count) + '.png')
        self.count= self.count + 1
        return img
################

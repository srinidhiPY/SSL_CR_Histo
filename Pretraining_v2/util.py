import numpy as np
from skimage import color
import openslide
import os
import glob


###############
def isforeground(self, scan, xpos, ypos, level, mu_percent=0.1, thresh=0.75):     # Mask Generation
    wsi = scan.read_region((xpos, ypos), level, (self.tile_w, self.tile_h)).convert('RGB')
    hsv = color.rgb2hsv(np.asarray(wsi))
    hsv = hsv[..., 1] > mu_percent
    return np.count_nonzero(hsv) / hsv.size >= thresh


###########
class AverageMeter(object):

    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#############

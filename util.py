import numpy as np
from skimage import color
from skimage.color import rgb2hed
from skimage.color import hed2rgb
import random
import h5py
import os
import glob
import copy
import json
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from skimage.measure import points_in_poly


def isforeground(self, scan, xpos, ypos, level, mu, mu_percent=0.15, thresh=0.95):     # Mask Generation
    wsi = scan.read_region((xpos, ypos), level, (self.tile_w, self.tile_h)).convert('RGB')
    lab = color.rgb2lab(np.asarray(wsi))
    lab = lab[..., 1] > (1 + mu_percent) * mu

    return np.count_nonzero(lab) / lab.size >= thresh


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


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __setitem__(self, key, value):
        self.__dict__.update({key: value})



############ Color Augmentation ###########################

def colour_augmentation(image, h_mean=0, h_std=random.uniform(-0.035, 0.035), d_mean=0, d_std=random.uniform(-0.035, 0.035), e_mean=0, e_std=random.uniform(-0.035, 0.035)):

    '''
    Randomly augments staining of images by separating them in to h and e (and d)
    channels and modifying their values. Aims to produce plausible stain variation
    used in custom augmentation

    PARAMETERS
    ##########

    image - arbitary RGB image (3 channel array) expected to be 8-bit

    aug_mean - average value added to each stain, default setting is 0

    aug_std - standard deviation for random modifier, default value 0.035

    RETURNS
    #######

    image - 8 bit RGB image with the same dimensions as the input image, with
            a modified stain
    ​
    '''

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

    # maskFlat = np.ravel(mask, order='A')

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

##########
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    return ax

######## Camelyon16 annotations #####

np.random.seed(0)

class Polygon(object):
    """
    Polygon represented as [N, 2] array of vertices
    """
    def __init__(self, name, vertices):
        """
        Initialize the polygon.
        Arguments:
            name: string, name of the polygon
            vertices: [N, 2] 2D numpy array of int
        """
        self._name = name
        self._vertices = vertices

    def __str__(self):
        return self._name

    def inside(self, coord):
        """
        Determine if a given coordinate is inside the polygon or not.
        Arguments:
            coord: 2 element tuple of int, e.g. (x, y)
        Returns:
            bool, if the coord is inside the polygon.
        """
        return points_in_poly([coord], self._vertices)[0]

    def vertices(self):

        return np.array(self._vertices)


class Annotation(object):
    """
    Annotation about the regions within WSI in terms of vertices of polygons.
    """
    def __init__(self):
        self._json_path = ''
        self._polygons_positive = []
        self._polygons_negative = []

    def __str__(self):
        return self._json_path

    def from_json(self, json_path):
        """
        Initialize the annotation from a json file.
        Arguments:
            json_path: string, path to the json annotation.
        """
        self._json_path = json_path
        with open(json_path) as f:
            annotations_json = json.load(f)

        for annotation in annotations_json['positive']:
            name = annotation['name']
            vertices = np.array(annotation['vertices'])
            polygon = Polygon(name, vertices)
            self._polygons_positive.append(polygon)

        for annotation in annotations_json['negative']:
            name = annotation['name']
            vertices = np.array(annotation['vertices'])
            polygon = Polygon(name, vertices)
            self._polygons_negative.append(polygon)

    def inside_polygons(self, coord, is_positive):
        """
        Determine if a given coordinate is inside the positive/negative
        polygons of the annotation.
        Arguments:
            coord: 2 element tuple of int, e.g. (x, y)
            is_positive: bool, inside positive or negative polygons.
        Returns:
            bool, if the coord is inside the positive/negative polygons of the
            annotation.
        """
        if is_positive:
            polygons = copy.deepcopy(self._polygons_positive)
        else:
            polygons = copy.deepcopy(self._polygons_negative)

        for polygon in polygons:
            if polygon.inside(coord):
                return True

        return False

    def polygon_vertices(self, is_positive):
        """
        Return the polygon represented as [N, 2] array of vertices
        Arguments:
            is_positive: bool, return positive or negative polygons.
        Returns:
            [N, 2] 2D array of int
        """
        if is_positive:
            return list(map(lambda x: x.vertices(), self._polygons_positive))
        else:
            return list(map(lambda x: x.vertices(), self._polygons_negative))


class Formatter(object):
    """
    Format converter e.g. CAMELYON16 to internal json
    """
    def camelyon16xml2json(inxml, outjson):
        """
        Convert an annotation of camelyon16 xml format into a json format.
        Arguments:
            inxml: string, path to the input camelyon16 xml format
            outjson: string, path to the output json format
        """
        root = ET.parse(inxml).getroot()
        annotations_tumor = \
            root.findall('./Annotations/Annotation[@PartOfGroup="Tumor"]')
        annotations_0 = \
            root.findall('./Annotations/Annotation[@PartOfGroup="_0"]')
        annotations_1 = \
            root.findall('./Annotations/Annotation[@PartOfGroup="_1"]')
        annotations_2 = \
            root.findall('./Annotations/Annotation[@PartOfGroup="_2"]')
        annotations_positive = \
            annotations_tumor + annotations_0 + annotations_1
        annotations_negative = annotations_2

        json_dict = {}
        json_dict['positive'] = []
        json_dict['negative'] = []

        for annotation in annotations_positive:
            X = list(map(lambda x: float(x.get('X')),
                     annotation.findall('./Coordinates/Coordinate')))
            Y = list(map(lambda x: float(x.get('Y')),
                     annotation.findall('./Coordinates/Coordinate')))
            vertices = np.round([X, Y]).astype(int).transpose().tolist()
            name = annotation.attrib['Name']
            json_dict['positive'].append({'name': name, 'vertices': vertices})

        for annotation in annotations_negative:
            X = list(map(lambda x: float(x.get('X')),
                     annotation.findall('./Coordinates/Coordinate')))
            Y = list(map(lambda x: float(x.get('Y')),
                     annotation.findall('./Coordinates/Coordinate')))
            vertices = np.round([X, Y]).astype(int).transpose().tolist()
            name = annotation.attrib['Name']
            json_dict['negative'].append({'name': name, 'vertices': vertices})

        with open(outjson, 'w') as f:
            json.dump(json_dict, f, indent=1)

    def vertices2json(outjson, positive_vertices=[], negative_vertices=[]):
        json_dict = {}
        json_dict['positive'] = []
        json_dict['negative'] = []

        for i in range(len(positive_vertices)):
            name = 'Annotation {}'.format(i)
            vertices = positive_vertices[i].astype(int).tolist()
            json_dict['positive'].append({'name': name, 'vertices': vertices})

        for i in range(len(negative_vertices)):
            name = 'Annotation {}'.format(i)
            vertices = negative_vertices[i].astype(int).tolist()
            json_dict['negative'].append({'name': name, 'vertices': vertices})

        with open(outjson, 'w') as f:
            json.dump(json_dict, f, indent=1)

###################################

import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob
import cv2

from torch.utils import data

def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors

    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0,0,0], [128,0,0], [0,128,0], [128,128,0],
                      [0,0,128], [128,0,128], [0,128,128], [128,128,128],
                      [64,0,0], [192,0,0], [64,128,0], [192,128,0],
                      [64,0,128], [192,0,128], [64,128,128], [192,128,128],
                      [0, 64,0], [128, 64, 0], [0,192,0], [128,192,0],
                      [0,64,128]])


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes

    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.

    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def decode_segmap(label_mask, plot=False):
    """Decode segmentation class labels into a color image

    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.

    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    label_colours = get_pascal_labels()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, 21):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r #/ 255.0
    rgb[:, :, 1] = g #/ 255.0
    rgb[:, :, 2] = b #/ 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

for i in range(1,21):
    if not os.path.exists('./fewshot/label/%s' % i):
        os.makedirs('./fewshot/label/%s' % i)
imgnames = os.listdir('./data/VOC2012/SegmentationClass')
cnt = 0
for name in imgnames:
    cnt += 1
    print ('%s / %s' % (cnt, len(imgnames)))
    img = cv2.imread('./data/VOC2012/SegmentationClass/%s' % name)
    img = img[:,:,::-1]
    img = cv2.resize(img, (224,224), interpolation=cv2.INTER_NEAREST)
    label = encode_segmap(img)
    clss = np.unique(label)
    clss = np.trim_zeros(clss)
    for c in clss:
        result = label.copy()
        result[result!=c] = 0
        result[result==c] = 1
        # print (np.unique(result))
        cv2.imwrite('./fewshot/label/%s/%s' % (c, name), result)
    jpegname = name.replace('.png', '.jpg')
    rgb = cv2.imread('./data/VOC2012/JPEGImages/%s' % jpegname)
    rgb = cv2.resize(rgb, (224,224))
    if rgb is None:
        print ('wrong here, stop')
    cv2.imwrite('./fewshot/image/%s' % jpegname, rgb)

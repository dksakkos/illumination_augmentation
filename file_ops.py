"""
Illumination-Based Data Augmentation for Robust Background Subtraction

Copyright (c) 2019 Dimitrios SAKKOS, Hubert SHUM and Edmond S. L. HO.
Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (see LICENSE for details)

Written by Dimitrios Sakkos
"""

import os, ntpath
from keras.preprocessing import image as kImage
import numpy as np
import skimage.io as sk
from glob import glob
from random import sample


def get_dataset_split(sample_size=None):
    img_train = glob(os.path.join('SABS', 'Darkening', '*.png'))
    img_train += glob(os.path.join('SABS', 'NoForegroundNight', '*.png'))
    img_train = [x for x in img_train if int(x[-8:-4]) < 800]
    img_train = [x + ',' + x.replace('.png', '_gt.png') for x in img_train]
    img_val = glob(os.path.join('SABS', 'LightSwitch', '*.png'))
    img_val = [x + ',' + x.replace('.png', '_gt.png') for x in img_val]
    if sample_size:
        sample_size = min(sample_size, len(img_val))
        img_val = list(sample(img_val, sample_size))
    return img_train, img_val


def read_image(img_path, pad=None):
    img = kImage.load_img(img_path)
    img = kImage.img_to_array(img)
    if pad: img = np.pad(img, [pad, (0, 0), (0, 0)], 'reflect')
    img = np.expand_dims(img, 0)
    assert (len(img.shape) == 4)
    return img


def read_gt(gt_path, pad=None):
    gt = sk.imread(os.path.join(gt_path, 'ground_truth/gt_bin', gt_path).replace('\n', ''), 0)
    if np.max(gt) == 255: gt = normalise_img(gt)
    if pad: gt = np.pad(gt, [pad, (0, 0)], 'constant', constant_values=0)
    assert (list(np.unique(gt)) in [[0], [0, 1]])
    gt = np.expand_dims(gt, 0)
    gt = np.expand_dims(gt, -1)
    return gt

def normalise_img(img):
    return img/255.

def create_folder(name):
    while not os.path.exists(name):
        os.makedirs(name)

import warnings

def save_image(img, name):
    if np.max(img) <= 1: img = img * 255
    img = img.astype(np.uint8)
    create_folder(ntpath.dirname(name))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sk.imsave(name, img)
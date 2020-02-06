"""
Illumination-Based Data Augmentation for Robust Background Subtraction

Copyright (c) 2019 Dimitrios SAKKOS, Hubert SHUM and Edmond S. L. HO.
Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (see LICENSE for details)

Written by Dimitrios Sakkos
"""

import numpy as np
import os, sys, json

# set current working directory
from augmenter import Augmenter
from file_ops import read_gt, read_image, get_dataset_split, create_folder, normalise_img
from utils import sort_imgs, matthews

workingpath = os.getcwd()
os.chdir(workingpath)
sys.path.append(workingpath)

from keras import backend as K, optimizers
import keras, ntpath
from sklearn.utils import compute_class_weight
from segmentation_models.metrics import iou_score, f1_score
from keras.callbacks import CSVLogger
from segmentation_models import Unet
from argparse import ArgumentParser

parser = ArgumentParser(description='Network configurations')

parser.add_argument(
    '--backbone', required=False, default='vgg16',
    choices=('vgg16', 'efficientnetb0','resnet34', 'mobilenetv2', 'seresnet34'),
    help='Pre-trained backbone')

parser.add_argument(
    '--augment_local', required=False, default=(120, 160), type=int, nargs=2,
    help='Minimum/maximum value of local changes mask')

parser.add_argument(
    '--augment_global', required=False, default=(40, 80), type=int, nargs=2,
    help='Minimum/maximum value of global changes mask')

parser.add_argument(
    '--augment_regular', action='store_true', default=False,
    help='Use the regular augmentor (image mirroring and noise)')

parser.add_argument(
    '--lr_decay', required=False, default=0.1, type=float,
    help='Learning rate decay factor')

parser.add_argument(
    '--lr', required=False, default=2e-3, type=float,
    help='Initial learning rate')

parser.add_argument(
    '--epochs', required=False, default=100, type=int,
    help='Maximum number of epochs')

parser.add_argument(
    '--reduce_patience', required=False, default=4, type=int,
    help='Optimiser patience: number of epochs of no improvement until we reduce the learning rate')

parser.add_argument(
    '--stop_patience', required=False, default=15, type=int,
    help='Optimiser patience: number of epochs of no improvement until we stop training')

parser.add_argument(
    '--name', required=False, default='mymodel',
    help='Name of the model')

parser.add_argument(
    '--epoch_size', required=False, default=500, type=int,
    help='Images per epoch')

parser.add_argument(
    '--val_size', required=False, default=100, type=int,
    help='The size of the validation set')

vgg_weights_path = os.path.join(ntpath.dirname(workingpath), 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

def get_class_weights(data, samples=100):
    gts = []
    for i in range(min(len(data), samples)):
        gt = read_gt(data[i].split(',')[1])
        if len(np.unique(gt)) == 1: continue
        gts.append(gt)
    cls_weight = compute_class_weight('balanced', [0, 1], np.asarray(gts).flatten())
    print('class weights:', cls_weight)
    return cls_weight

def data_generator(file_list, augmenter = None, sort_data=False):
    if len(file_list) == 0: raise Exception("Empty data list")
    if sort_data:
        file_list = sort_imgs(file_list)
    else:
        np.random.shuffle(file_list)
    counter = 0
    while True:
        current_img, current_gt = file_list[counter].split(',')
        img = read_image(current_img, pad=(4, 4))
        gt = read_gt(current_gt, pad=(4, 4)) if 'noforeground' not in current_gt.lower() else np.zeros_like(
            img[..., 0:1])
        if augmenter: img, gt = augmenter.augment_image(img, gt)
        img = normalise_img(img)
        counter += 1

        if counter == len(file_list) - 1:
            counter = 0
            if not sort_data: np.random.shuffle(file_list)
        yield (img, gt)


def get_callback_list():
    chk = keras.callbacks.ModelCheckpoint(os.path.join(workingpath, args.name, args.name +'.h5'),
                                          monitor='val_score', verbose=1, save_best_only=True,
                                          save_weights_only=False, mode='max', period=1)

    redu = keras.callbacks.ReduceLROnPlateau(monitor='val_score', factor=args.lr_decay,
                                             patience=args.reduce_patience, min_lr=1e-7, verbose=1, mode='max')

    early = keras.callbacks.EarlyStopping(monitor='val_score', min_delta=1e-4, patience=args.stop_patience,
                                          verbose=0, mode='max')

    csv_logger = CSVLogger(os.path.join(workingpath, args.name, args.name+'.csv'), append=True,
                           separator=',')
    return [chk, redu, early, csv_logger]


args = parser.parse_args()
create_folder(args.name)
with open(os.path.join(workingpath, args.name, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

model = Unet(args.backbone, encoder_freeze=True, encoder_weights='imagenet')
metrics_list = [iou_score, f1_score, matthews]
opt = optimizers.adam(args.lr)
model.compile(loss=K.binary_crossentropy, metrics=metrics_list, optimizer=opt)
model.summary()

augmenter = Augmenter(local_mask=args.augment_local, global_mask=args.augment_global,
                      flip_and_noise=args.augment_regular)

img_train, img_val = get_dataset_split(sample_size=150)

model.fit_generator(data_generator(img_train, sort_data=False, augmenter=augmenter),
                    steps_per_epoch=args.epoch_size, epochs=args.epochs, callbacks=get_callback_list(),
                    class_weight=get_class_weights(img_train),
                    validation_data=data_generator(img_val, sort_data=False),
                    validation_steps=args.val_size)

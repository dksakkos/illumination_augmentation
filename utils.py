"""
Illumination-Based Data Augmentation for Robust Background Subtraction

Copyright (c) 2019 Dimitrios SAKKOS, Hubert SHUM and Edmond S. L. HO.
Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (see LICENSE for details)

Written by Dimitrios Sakkos
"""

import re
import keras.backend as K
from segmentation_models.metrics import iou_score, f1_score

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def sort_imgs(img_path_list):
    img_path_list.sort(key=natural_keys)
    return img_path_list

def get_custom_objects():
    return {
        'matthews': matthews,
        'score': f1_score,
        'iou_score': iou_score,
    }

def printMetrics(metrics):
    names = ['Recall', 'Sp', 'FPR', 'FNR', 'PWC', 'FM', 'Precision', 'IoU', 'Matthews']
    for i in range(len(names)):
        print(names[i], metrics[i])

def display_progress(number):
    print("Progress: {:.2f}%".format(number * 100.0)
          , end="\r", flush=True)


def get_data_list(path):
    data = open(path, 'r').readlines()
    return data

def matthews(y_true, y_pred):
    y_true = K.flatten(y_true[...,0])
    y_pred = K.flatten(y_pred)
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())
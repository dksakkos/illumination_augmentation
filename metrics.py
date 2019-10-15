"""
Illumination-Based Data Augmentation for Robust Background Subtraction

Copyright (c) 2019 Dimitrios SAKKOS, Hubert SHUM and Edmond S. L. HO.
Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (see LICENSE for details)

Written by Dimitrios Sakkos
"""

import numpy as np
import keras.backend as K

def getMetrics(classifications):
    TP, FP, FN, TN = classifications.astype(np.float64)
    try:
        Recall = TP * 1.0 / (TP + FN) * 1.0
        Precision = TP * 1.0 / (TP + FP) * 1.0
        Sp = TN * 1.0 / (TN + FP) * 1.0
        FPR = FP * 1.0 / (FP + TN) * 1.0
        FNR = FN * 1.0 / (TP + FN) * 1.0
        PWC = 100.0 * (FN + FP) / (TP + FN + FP + TN) * 1.0
        FM = (2 * Precision * Recall) / (Precision + Recall)
        IoU = TP * 1.0 / (TP + FP + FN) * 1.0
        numerator = (TP * TN - FP * FN)
        denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        Matthews= numerator / (denominator + 1e-5)
    except ZeroDivisionError as e:
        return [float('NaN')]
    return [Recall, Sp, FPR, FNR, PWC, FM, Precision, IoU, Matthews]

def getPixels(out, gt, threshold):

    out[out>threshold]=1
    out[out<=threshold]=0

    gt = gt.flatten()
    out = out.flatten()
    unq_out = np.unique(out)
    unq_gt = np.unique(gt)
    if len(unq_out)==1: assert(unq_out==[0])
    else: assert((unq_out==[0,1]).all())
    if len(unq_gt)==1: assert(unq_gt==[0])
    else: assert((unq_gt==[0,1]).all())
    gt[gt==-1]=50


    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    tp = np.sum(np.logical_and(out == 1, gt == 1))

    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    tn = np.sum(np.logical_and(out == 0, gt == 0))

    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    fp = np.sum(np.logical_and(out == 1, gt == 0))

    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    fn = np.sum(np.logical_and(out == 0, gt == 1))

    im_shape = [i for i in gt.shape]
    total_pixels = 1
    for pix in im_shape:
        total_pixels = total_pixels * pix

    assert(tp+tn+fp+fn==pix)

    return np.array([tp,fp,fn,tn])
    
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

"""
Illumination-Based Data Augmentation for Robust Background Subtraction

Copyright (c) 2019 Dimitrios SAKKOS, Hubert SHUM and Edmond S. L. HO.
Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (see LICENSE for details)

Written by Dimitrios Sakkos
"""

from file_ops import read_image, read_gt, get_dataset_split, save_image, normalise_img
import keras, ntpath, os
import numpy as np

from metrics import getMetrics, getPixels
from utils import get_custom_objects, printMetrics, display_progress

from argparse import ArgumentParser

parser = ArgumentParser(description='Testing options')

parser.add_argument(
    '--model_name', required=True,
    help='The path to the parent folder of the trained model')

parser.add_argument(
    '--threshold', required=False, default=0.5,
    help='The path to the trained model')

parser.add_argument(
    '--save_results', required=False, default=False,
    help='The path to the trained model')

parser.add_argument(
    '--verbose', required=False, default=True,
    help='The path to the trained model')

def test(model_name, threshold=0.5, save=True, verbose=True):
    classifications = np.array([0, 0, 0, 0])
    results_folder = os.path.join(model_name, 'results')
    _, test_set = get_dataset_split()
    model = keras.models.load_model(
        os.path.join(model_name, model_name + '.h5'), custom_objects=get_custom_objects())
    for i in range(len(test_set)):
        if verbose: display_progress(i / len(test_set))
        img_path, gt_path = test_set[i].replace('\n', '').split(',')

        img = read_image(img_path, pad=(4, 4))
        img = normalise_img(img)
        ground_truth = read_gt(gt_path, pad=(4, 4))

        prediction = model.predict(img)

        prediction = np.squeeze(prediction)
        ground_truth = np.squeeze(ground_truth)
        prediction = prediction[4:-4, ...]
        ground_truth = ground_truth[4:-4, ...]

        prediction = (prediction > threshold).astype(np.uint8)

        classifications += getPixels(prediction, ground_truth, 0.5)

        if save:
            save_image(prediction, os.path.join(results_folder, ntpath.basename(img_path)))
            save_image(ground_truth, os.path.join(results_folder,
                                              ntpath.basename(img_path).replace('.png', '_gt.png')))


    print(model_name, threshold)
    printMetrics(getMetrics(classifications))

args = parser.parse_args()
test(args.model_name, threshold=float(args.threshold), save=args.save_results, verbose=bool(args.verbose))

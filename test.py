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
from utils import get_custom_objects, printMetrics, display_progress, sort_imgs, create_map, get_prob_map

from argparse import ArgumentParser

parser = ArgumentParser(description='Testing options')

parser.add_argument(
    '--name', required=True,
    help='The path to the parent folder of the trained model')

parser.add_argument(
    '--threshold', required=False, default=0.5,
    help='The path to the trained model')

parser.add_argument(
    '--save_results', action='store_true', default=False,
    help='The path to the trained model')

parser.add_argument(
    '--verbose', action='store_true', default=True,
    help='The path to the trained model')

parser.add_argument(
    '--refine', action='store_true', default=False,
    help='Enables the post-processing method')

def test(model_name, threshold=0.5, save=True, verbose=True, refine=False):
    classifications = np.array([0, 0, 0, 0])
    results_folder = os.path.join(model_name, 'results')
    if not os.path.exists(results_folder): os.mkdir(results_folder)
    _, test_set = get_dataset_split()
    if refine: test_set = sort_imgs(test_set)
    prediction = None
    model = keras.models.load_model(
        os.path.join(model_name, model_name + '.h5'), custom_objects=get_custom_objects())
    for i in range(len(test_set)):
        if verbose: display_progress(i / len(test_set))
        img_path, gt_path = test_set[i].replace('\n', '').split(',')

        img = read_image(img_path, pad=(4, 4))
        img = normalise_img(img)
        ground_truth = read_gt(gt_path)
        ground_truth = np.squeeze(ground_truth)

        if refine:
            prediction = ground_truth if prediction is None else prediction
            pmap = create_map(prediction > 0.5, 1)
            prob = get_prob_map(pmap)

        prediction = model.predict(img)

        prediction = np.squeeze(prediction)
        prediction = prediction[4:-4, ...]

        prediction = (prediction > threshold).astype(np.uint8)
        if refine: prediction = prediction * prob

        classifications += getPixels(prediction, ground_truth, 0.5)

        if save:
            save_image(prediction, os.path.join(results_folder, ntpath.basename(img_path)))
            save_image(ground_truth, os.path.join(results_folder,
                                              ntpath.basename(img_path).replace('.png', '_gt.png')))
#            if refine:
#                prob = prob.astype(np.uint8)
#                save_image(os.path.join(results_folder,
#                                ntpath.basename(img_path).replace('.png', '_prob.png')), prob)

    print(model_name, threshold)
    printMetrics(getMetrics(classifications))

args = parser.parse_args()
print(args)
test(args.name, threshold=float(args.threshold), save=args.save_results,
     verbose=bool(args.verbose), refine=bool(args.refine))

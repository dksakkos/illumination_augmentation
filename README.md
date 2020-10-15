# Illumination-Based Data Augmentation for Robust Background Subtraction

This repository contains the Keras code used in the paper [Illumination-Based Data Augmentation for Robust Background Subtraction](https://ieeexplore.ieee.org/document/8982527),
which received the **Best paper award** at [**SKIMA 2019**](http://skimanetwork.info/) which was held from 26 to 28 August 2019 in Island of Ulkulhas, Maldives. 

An extension of this work was published at the **Journal of Enterprise Information Management** as [Image editing-based data augmentation for illumination-insensitive background subtraction](https://www.emerald.com/insight/content/doi/10.1108/JEIM-02-2020-0042/full/html).

# SABS dataset
The dataset used in this study was developed by [Stuttgart University](https://www.vis.uni-stuttgart.de/forschung/visual-analytics/visuelle-analyse-videostroeme/stuttgart_artificial_background_subtraction_dataset/).
For training the models the *Darkening* video sequence was used, while the test set consists of the *LightSwitch* video.

# Training

Once the dataset has been downloaded, it needs to be extracted in the same directory as the source code files and placed in a folder named `SABS`, which should contain the *Darkening* and *LightSwitch* subfolders.

To use the VGG16 pre-trained backbone, you can download the weights from [this link](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5) and save the file in the working directory.

You are then ready to run the `train.py` script to start training, which accepts various self-explanatory arguments. All of these have pre-determined values for your convenience.
To use the proposed illumination augmenter, provide the *augment_local* and *augment_global* arguments followed by the minimum and maximum pixel values of the mask intensity.
For example, to train a model with a VGG16 backbone and the proposed augmentation method, run the script as follows:

```
python train.py \
    --name vgg16_augmented \
    --backbone vgg16 \
    --augment_local 120 160 \
    --augment_global 40 80
```

The training arguments will be saved at your working folder, to ease the tracking of your experiments. To turn off the proposed method and use the regular augmentor instead, run the following:

```
python train.py \
    --augment_local 0 0 \
    --augment_global 0 0 \
    --augment_regular
```


# Testing
For evaluating a trained model, you need to provide the filename of the trained model.
Optionally, you can save the predictions, set a different binarisation threshold and control the depiction of a progress bar as follows:

```
python test.py \
    --name vgg16_augmented \
    --threshold 0.7 \
    --save_results
```

You can also refine the results with the proposed post-processing method, using the argument ```--refine```.

# Notes
During training, the script will monitor, for each epoch, the F1 score on the validation set and overwrite the saved model in case of improvement.
Training stops after a number of consecutive epochs of no improvement, which is pre-set as 15.

## Citation

If you use this code in your research, please use the following BibTeX entries.

````
@article{sakkos_shum_ho_2019,
title={Illumination-Based Data Augmentation for Robust Background Subtraction},
DOI={10.1109/skima47702.2019.8982527},
journal={2019 13th International Conference on Software, Knowledge, Information Management and Applications (SKIMA)}, 
author={Sakkos, Dimitrios and Shum, Hubert P. H. and Ho, Edmond S. L.},
year={2019}
}

@article{sakkos_ho_shum_elvin_2020,
title={Image editing-based data augmentation for illumination-insensitive background subtraction},
DOI={10.1108/jeim-02-2020-0042},
journal={Journal of Enterprise Information Management},
author={Sakkos, Dimitrios and Ho, Edmond S. L. and Shum, Hubert P. H. and Elvin, Garry},
year={2020}
}
```` 

## Contact
For any questions or discussions, you can contact me at dksakkos@gmail.com.

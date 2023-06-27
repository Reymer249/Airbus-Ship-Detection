# Airbus-Ship-Detection
This is the repository for [Airbus Ship Detection Challenge](https://www.kaggle.com/competitions/airbus-ship-detection) on Kaggle. In this competition, you are required to locate ships in images and put an aligned bounding box segment around the ships you locate. However, in this implementation the technique to output the result is segmentation: the area where the ships are supposed to be according to the model will be highlighted. \
The original notebook with all elaboration may be found in the repository (model.jpynb) or [here](https://colab.research.google.com/drive/1BscNdmSV3YH0AipWZXroufdcCCdVNdfE?usp=sharing)

## Solution
For this problem, U-net architecture with some additional layers (Gaussian noise, batch normalization) was used to build a model.  During the work, it was noticed, that the set is extremely unbalanced into the side of the empty images. Therefore, undersampling was done. Also, most of the ships occupy a tiny space in the pictures (2-5%). This fact turned out to be a small issue, as the network tended to highlight even some small noises (like wawes, barrels, waste and so on). You may see that in the example of the prediction (_prediction_sample.png_). In the code, it was fixed using the small processing during the output generation (using scikit-image). Apart from that, another solution arose. I was trying to build another model, which will just classify the images into ‘with ships’ and ‘empty’. After that, the images labelled as ‘with ships’ would be fed into the main model. Such cooperation of networks may increase the accuracy of the prediction. The classification model is finished (_supplements_ folder) but wasn’t trained. To be more precise, it was trained only for 5 epochs, which is not enough to produce a proper result. So the work on the connection of these models is still in progress…

## Files
* data - folder with the data for the model
  - test - this folder contains the pictures you want to put into a model
* supplements - folder with the additional materials, read about it in the section below
  - class_model.jpynb - a notebook with the elaboration of the classification model
  - class_model_1.h5 - a file containing the classification model
  - class_model_1_fullres.h5 - a file containing the classification model with the image scaling inside the model
  - clas_model_weights.best.hdf5 - a file containing the best weights of the classification model **(irrelevant, only 5 epochs)**
* README.md - the file you are reading
* fullres_model.h5 - a file containing the model with the image scaling inside the model
* model.jpynb - a notebook with an elaboration of the model (also [here](https://colab.research.google.com/drive/1BscNdmSV3YH0AipWZXroufdcCCdVNdfE?usp=sharing))
* model_inference.py - script, which will make the model produce the output (output.csv) for the images inside the _data/test_ folder
* output.csv - example of the output with only one image (in the _data/test_ folder)
* prediction_sample.png - the image with the examples of the predictions model outputs
* requirements.txt - file with the names of Python libraries needed for proper work of the model
* seg_model.h5 - the file containing the model
* seg_model_weights.best.hdf5 - the file containing the weights for the model
* train_model.py - script, which will run the training of the model (**Note**, that for that there should be the training pictures inside the _data/train_ folder and a file, named _train_ship_segmentations_v2.csv_, with RLE masks of the ships inside a general folder)

## Making predictions
To run the network and receive prediciton follow these steps:

1. Put all the images you want to analyze into the _data/test_ folder
2. Make sure, that you have all the modules listed in the _requirements.txt_ file
3. Run* the _model_inference.py_ file
4. The file, named output.csv will be created. It is ht prediction of the model.

 \***Note:** If you are facing the issue with tqdm module whilst running the program, try to change the following on the 101 line of code:

```
for c_img_name in tqdm(test_paths):
```

into

```
for c_img_name in test_paths:
```

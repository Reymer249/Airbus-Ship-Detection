"""# Import and constants"""

import numpy as np
import pandas as pd
from skimage.io import imread
import os
from keras import models, layers
from tqdm.notebook import tqdm # for the nice visualisation
from skimage.morphology import binary_opening, disk # final results processing
import gc; gc.enable()

directory_path = "data"
train_path = os.path.join(directory_path, 'train')
test_path = os.path.join(directory_path, 'test')

BATCH_SIZE = 4
EDGE_CROP = 16
# Number of epochs
NB_EPOCHS = 150
# The coefficient of Gaussian noise added
# on the 1st layer of the CNN
GAUSSIAN_NOISE = 0.1
# Downsampling inside the network
# If you want to scale down the image,
# please input tuple with the resolution
NET_SCALING = None
# Downsampling (scaling) in preprocessing
IMG_SCALING = (1, 1)
# Number of validation images to use
VALID_IMG_COUNT = 600
# Maximum number of steps per epoch in training
MAX_TRAIN_STEPS = 200

"""# Needed encoding/decoding functions

The functions for Run-Length decoding and encoding
"""

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

# This one will be needed further to decode pixels into an image.
# It decodes ALL masks of the image in one go.
def masks_as_image(in_mask_list: np.array):
    '''
    Takes the list of the masks (ships) and create a single mask array for all ships.
    Returns numpy array, in which 0 element is a needed encoding (1 - mask, 0 - background)
    '''
    all_masks = np.zeros((768, 768), dtype = np.int16)
    for mask in in_mask_list: # going throug all masks
        if isinstance(mask, str):
            # if mask in str type, then decode and add
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)


"""# Importing the model
"""

if not 'seg_model' in globals():
  seg_model = models.load_model('seg_model.h5', compile=False)
  seg_model.load_weights('seg_model_weights.best.hdf5')


"""# Submission

This section deals with the submission of the results for the competition
"""

test_paths = os.listdir(test_path)
out_pred_rows = []

# If tqdm doesn't want to work, then just change:
# tqdm(test_paths) -> test_paths
# This was a solution on my machine.
for c_img_name in tqdm(test_paths):
    c_path = os.path.join(test_path, c_img_name)
    c_img = imread(c_path)
    c_img = np.expand_dims(c_img, 0)/255.0
    cur_seg = seg_model.predict(c_img)[0]
    cur_seg = binary_opening(cur_seg>0.5, np.expand_dims(disk(2), -1))
    cur_rles = rle_encode(cur_seg)
    if len(cur_rles)>0:
            out_pred_rows += [{'ImageId': c_img_name, 'EncodedPixels': cur_rles}]
    else:
        out_pred_rows += [{'ImageId': c_img_name, 'EncodedPixels': None}]
    gc.collect()

submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
submission_df.to_csv('output.csv', index=False)

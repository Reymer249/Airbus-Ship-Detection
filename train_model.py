"""Import"""

import numpy as np
import pandas as pd
from skimage.io import imread
import matplotlib.pyplot as plt
import os
import gc; gc.enable()

directory_path = "data"
train_path = os.path.join(directory_path, 'train')
test_path = os.path.join(directory_path, 'test')

'''CONSTANTS'''

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

"""# Importing data"""

masks = pd.read_csv(os.path.join(directory_path, 'train_ship_segmentations_v2.csv'))

"""# Splitting into train and validation set"""

# creating a new column with the indication thether there are ship on the picture (1) or no (0)
masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
# creating a new DataFrame, in which information is grouped by the id of the image
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
# we create a new column with the 1 if image has ships on it and 0 otherwise
unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x]) # same as prev. but list (vector)
# filter files that are too small/corrupt
unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id:
                                                               os.stat(os.path.join(train_path,
                                                                                    c_img_id)).st_size/1024)

unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb']>50] # keep only 50kb files
# That's because if file weights less than 50, it is probably corrupted
masks.drop(['ships'], axis=1, inplace=True)

# Stratified sampling

from sklearn.model_selection import train_test_split

# Stratified sampling based on the number of ships
# Note, that we are sampling the id's of the images
# and not the masks themselve. That's because
# every mask is one ship, and if we sample basing
# on ships data will be messed up
train_ids, valid_ids = train_test_split(unique_img_ids,
                 test_size = 0.2,
                 stratify = unique_img_ids['ships'])

# JOIN with the previous dataframes
train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)

"""# Undersample 0-images (no ships)

In the dataset we have to much samples with no ships. Therefore, we undersample it.\
Note, that I forgot to change the rate of undersampling from 10 to smaller number (3-4) in the row:


```
return in_df.sample(base_rep_val//10)
```

But it did not affected the network significantly, so it's fine :)\
However, it **was changed** in the .py file with for training the model.
"""

train_df['strat_group'] = train_df['ships'].map(lambda x: (x+1)//2).clip(0, 7)

def sample_ships(in_df, base_rep_val=1500):
    if in_df['ships'].iloc[0] == 0:
        return in_df.sample(base_rep_val//3) # even more strongly undersample no ships
    else:
        return in_df.sample(base_rep_val, replace=(in_df.shape[0]<base_rep_val))

balanced_train_df = train_df.groupby('strat_group').apply(sample_ships)


"""# Decoding RLE into images (arrays)

Creating a generator for the images we will feed into the CNN
"""

def make_image_gen(in_df, batch_size = BATCH_SIZE):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches: # current image, current masks
            rgb_path = os.path.join(train_path, c_img_id)
            c_img = imread(rgb_path)
            c_mask = masks_as_image(c_masks['EncodedPixels'].values)
            # if we want to scale the images during the preprocessing
            if IMG_SCALING is not None:
                c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]
                c_mask = c_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []

"""## Train images

Applying this generator to the train images to check whether everything is working properly
"""

mask_image_generator = make_image_gen(balanced_train_df)
train_x, train_y = next(mask_image_generator)

"""
## Validation images

Applying it to the validation images. Note, that as well as batch size we may also controle the number of images on which we will do the validation (VALID_IMG_COUNT)
"""

valid_x, valid_y = next(make_image_gen(valid_df, VALID_IMG_COUNT))

"""# Data augmentation

We expand our dataset using Keras (Tensorflow) backend
"""

from keras.preprocessing.image import ImageDataGenerator


dg_args = dict(featurewise_center = False,
                  samplewise_center = False,
                  rotation_range = 15,
                  width_shift_range = 0.1,
                  height_shift_range = 0.1,
                  shear_range = 0.01,
                  zoom_range = [0.9, 1.25],
                  horizontal_flip = True,
                  vertical_flip = True,
                  fill_mode = 'reflect',
                   data_format = 'channels_last')

image_gen = ImageDataGenerator(**dg_args)
label_gen = ImageDataGenerator(**dg_args)

def create_aug_gen(in_gen, seed = None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(255*in_x,
                             batch_size = in_x.shape[0],
                             seed = seed,
                             shuffle=True)
        g_y = label_gen.flow(in_y,
                             batch_size = in_x.shape[0],
                             seed = seed,
                             shuffle=True)

        yield next(g_x)/255.0, next(g_y)

full_gen = create_aug_gen(mask_image_generator)
train_full_x, train_full_y = next(full_gen)

gc.collect()

"""# Model

The U-net model architecture
"""

from keras import models, layers


def upsample(filters, kernel_size, strides, padding):
    return layers.Conv2DTranspose(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding)


input_img = layers.Input(train_full_x.shape[1:], name = 'RGB_Input')
pp_in_layer = input_img

# This option if we want to scale image inside the model itself.
# May be useful if we have images of different sizes and don't
# want to do the preprocessing.
# However, in the competition dataset every image of the same size,
# so we don't really need it for now.
if NET_SCALING is not None:
    pp_in_layer = layers.AvgPool2D(NET_SCALING)(pp_in_layer)

# Gaussian noise and batch normalization may help to regulize
# the data and improve predictions
pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(pp_in_layer)
pp_in_layer = layers.BatchNormalization()(pp_in_layer)

c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (pp_in_layer)
c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = layers.MaxPooling2D((2, 2)) (c1)

c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = layers.MaxPooling2D((2, 2)) (c2)

c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = layers.MaxPooling2D((2, 2)) (c3)

c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
p4 = layers.MaxPooling2D(pool_size=(2, 2)) (c4)

# -------------- End of the downsampling

c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

# -------------- Start of the upsampling

u6 = upsample(64, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = layers.concatenate([u6, c4])
c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

u7 = upsample(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = layers.concatenate([u7, c3])
c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

u8 = upsample(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = layers.concatenate([u8, c2])
c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

u9 = upsample(8, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = layers.concatenate([u9, c1], axis=3)
c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

d = layers.Conv2D(1, (1, 1), activation='sigmoid') (c9)
d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(d)
if NET_SCALING is not None:
    d = layers.UpSampling2D(NET_SCALING)(d)

seg_model = models.Model(inputs=[input_img], outputs=[d])
seg_model.summary()

"""## Monitoring of the training"""

import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
import tensorflow as tf

# function that computes the dice coefficient
def dice_coef(y_true, y_pred, smooth=1):
    y_tr = tf.cast(y_true, tf.float32)
    intersection = K.sum(y_tr * y_pred, axis=[1,2,3])
    union = K.sum(y_tr, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

# The combined (by binary crossentropy and dice coefficient) loss function we
# will use.  We aim to capture both the localization accuracy of the segmentation
# masks (through dice coefficient) and the overall pixel-wise classification
# accuracy (through binary cross-entropy).
def dice_p_bce(in_gt, in_pred):
    return 1e-3*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)

def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)

seg_model.compile(optimizer=Adam(1e-4, decay=1e-6), loss=dice_p_bce, metrics=[dice_coef, 'binary_accuracy', true_positive_rate])

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# The file we will save the best weight in
weight_path="{}_weights.best.hdf5".format('seg_model')

# This callback will save the best weights on each step
checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1,
                             save_best_only=True, mode='max', save_weights_only = True)

# This callback will reduce the learning rate when we are entering plateau.
# Such a atrategy may improve mdoel predictions
reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5,
                                   patience=3,
                                   verbose=1, mode='max', min_delta=0.0001, cooldown=2, min_lr=1e-6)

# This callback will stop the network training if we won't have any improvements
early = EarlyStopping(monitor="val_dice_coef",
                      mode="max",
                      patience=25)

callbacks_list = [checkpoint, early, reduceLROnPlat]

"""## Let's GO (training)"""

step_count = min(MAX_TRAIN_STEPS, balanced_train_df.shape[0]//BATCH_SIZE)
aug_gen = create_aug_gen(make_image_gen(balanced_train_df))

loss_history = [seg_model.fit(aug_gen,
                             steps_per_epoch=step_count,
                             epochs=NB_EPOCHS,
                             validation_data=(valid_x, valid_y),
                             callbacks=callbacks_list,
                             workers=1
                                       )]

"""## Results of the training"""

def show_loss(loss_history):
    epich = np.cumsum(np.concatenate(
        [np.linspace(0.5, 1, len(mh.epoch)) for mh in loss_history]))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 10))
    _ = ax1.plot(epich,
                 np.concatenate([mh.history['loss'] for mh in loss_history]),
                 'b-',
                 epich, np.concatenate(
            [mh.history['val_loss'] for mh in loss_history]), 'r-')
    ax1.legend(['Training', 'Validation'])
    ax1.set_title('Loss')

    _ = ax2.plot(epich, np.concatenate(
        [mh.history['true_positive_rate'] for mh in loss_history]), 'b-',
                     epich, np.concatenate(
            [mh.history['val_true_positive_rate'] for mh in loss_history]),
                     'r-')
    ax2.legend(['Training', 'Validation'])
    ax2.set_title('True Positive Rate\n(Positive Accuracy)')

    _ = ax3.plot(epich, np.concatenate(
        [mh.history['binary_accuracy'] for mh in loss_history]), 'b-',
                     epich, np.concatenate(
            [mh.history['val_binary_accuracy'] for mh in loss_history]),
                     'r-')
    ax3.legend(['Training', 'Validation'])
    ax3.set_title('Binary Accuracy (%)')

    _ = ax4.plot(epich, np.concatenate(
        [mh.history['dice_coef'] for mh in loss_history]), 'b-',
                     epich, np.concatenate(
            [mh.history['val_dice_coef'] for mh in loss_history]),
                     'r-')
    ax4.legend(['Training', 'Validation'])
    ax4.set_title('DICE')

    fig.savefig('training_results.png')

show_loss(loss_history)

"""## Saving"""

seg_model.load_weights(weight_path)
seg_model.save('seg_model.h5')

"""### and any resolution model
In case we want to upload images of some other resolution. That way we will have scaling right inside the model
"""

if IMG_SCALING is not None:
    fullres_model = models.Sequential()
    fullres_model.add(layers.AvgPool2D(IMG_SCALING, input_shape = (None, None, 3)))
    fullres_model.add(seg_model)
    fullres_model.add(layers.UpSampling2D(IMG_SCALING))
else:
    fullres_model = seg_model

fullres_model.save('fullres_model.h5')


"""# Valuation

Just a small final validation to take a look how the model works
"""


gc.collect()

final_valid_x, final_valid_y = valid_x, valid_y

final_valid_prediction = seg_model.predict(final_valid_x)

fig, m_axs = plt.subplots(8, 2, figsize = (10, 40))

for (ax1, ax2), image, prediction in zip(m_axs, final_valid_x, final_valid_prediction):
    ax1.imshow(image)
    ax1.set_title('Image')
    ax2.imshow(prediction)
    ax2.set_title('Prediction')

fig.savefig('predictions_sample.png')

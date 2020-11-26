import keras
from sklearn.model_selection import train_test_split

TEST_DIR = 'testing/'
import sys

SIGNATURE_CLASSES = ['A', 'B', 'C', 'D']
argument_data = sys.argv
if (len(argument_data) < 2):
    print('Insert name as commandline argument')
    sys.exit(0)
import os, random
import numpy as np
import pandas as pd
import cv2
import imageio
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from matplotlib import ticker

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, Activation
from keras.optimizers import SGD, Adagrad
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adam
from keras import backend as K

ROWS = 190
COLS = 160
CHANNELS = 3
TRAIN_DIR = "train/"


def root_mean_squared_error(y_true, y_pred):
    """
    RMSE loss function
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def get_images(fish):
    """Load files from train folder"""
    fish_dir = TRAIN_DIR + '{}'.format(fish)
    images = [fish + '/' + im for im in os.listdir(fish_dir)]
    # print(images)
    return images


def read_image(src):
    import os
    from scipy import misc
    filepath = src
    image = cv2.imread(filepath)
    import scipy.misc as mc
    scale_percent = 100  # percent of original size
    width = int(160)
    height = int(190)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


files = []
y_all = []

for fish in SIGNATURE_CLASSES:
    fish_files = get_images(fish)
    files.extend(fish_files)

    y_fish = np.tile(fish, len(fish_files))
    y_all.extend(y_fish)
    print("{0} photos of {1}".format(len(fish_files), fish))

y_all = np.array(y_all)
print(len(files))
print(len(y_all))

X_all = np.ndarray((len(files), ROWS, COLS, CHANNELS), dtype=np.uint8)

for i, im in enumerate(files):
    X_all[i] = read_image(TRAIN_DIR + im)
    cv2.imshow('img', X_all[i])
    cv2.waitKey(0)
    if i % 1000 == 0: print('Processed {} of {}'.format(i, len(files)))

print('X_all.shape')
print(X_all.shape)
# One Hot Encoding Labels
y_all = LabelEncoder().fit_transform(y_all)
y_all = np_utils.to_categorical(y_all)

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all,
                                                      test_size=4, random_state=23,
                                                      stratify=y_all)

optimizer = RMSprop(lr=1e-4)
objective = 'categorical_crossentropy'


def center_normalize(x):
    return (x - K.mean(x)) / K.std(x)


from keras.layers.convolutional import Conv2D

model = Sequential()
print('1')
model.add(Activation(activation=center_normalize, input_shape=(ROWS, COLS, CHANNELS)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(96, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(96, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (2, 2), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (2, 2), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(len(SIGNATURE_CLASSES)))
model.add(Activation('sigmoid'))

adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss=root_mean_squared_error)

early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')

model.fit(X_train, y_train, batch_size=64, epochs=200,
          validation_split=0.1, verbose=1, shuffle=True, callbacks=[early_stopping])
preds = model.predict(X_valid, verbose=1)
print("Validation Log Loss: {}".format(log_loss(y_valid, preds)))

test_files = [1]

test_files[0] = 'udara.jpeg'

test = np.ndarray((1, ROWS, COLS, CHANNELS), dtype=np.uint8)

print(TEST_DIR)
test[0] = read_image(TEST_DIR + argument_data[1] + '.jpeg')
print(TEST_DIR + im)

test_preds = model.predict(test, verbose=1)
print(test_preds)

submission = pd.DataFrame(test_preds, columns=SIGNATURE_CLASSES)
submission.insert(0, 'image', test_files)
submission.head()
print(submission)
submission.to_csv('signatureResults.csv', index=False)

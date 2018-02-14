import numpy as np
import pandas as pd
import bcolz
import time
from bcolzutils import *
from util import *
import logging
import datetime


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)


percent = 1
#percent = 0.005
d = datetime.datetime.today()
log_filename=f"train_topmodel_bottleneck_mobilenet_{d.year}-{d.month}-{d.day}-{d.hour}.{d.minute}.{d.second}.log"

logging.basicConfig(level='DEBUG', filename= log_filename)
log = logging.getLogger(__name__)
log.debug("fit and save mode using bottleneck features")

num_classes = 3
basedir="/media/hdd/datastore/t4sa"

bnf_valid_name = basedir + '/bottleneck_features_mobilenet_valid'
bnf_test_name = basedir + '/bottleneck_features_mobilenet_test'
bnf_train_name = basedir + '/bottleneck_features_mobilenet_train'

## Read it back from disk and check size
bnf_valid_data = bcolz.carray(rootdir=f'{bnf_valid_name}_data.bclz', mode='r')
bnf_test_data = bcolz.carray(rootdir=f'{bnf_test_name}_data.bclz', mode='r')
bnf_train_data = bcolz.carray(rootdir=f'{bnf_train_name}_data.bclz', mode='r')

bnf_valid_labels = bcolz.carray(rootdir=f'{bnf_valid_name}_labels.bclz', mode='r')
bnf_test_labels = bcolz.carray(rootdir=f'{bnf_test_name}_labels.bclz', mode='r')
bnf_train_labels = bcolz.carray(rootdir=f'{bnf_train_name}_labels.bclz', mode='r')

log.debug(bnf_valid_data.shape)
log.debug(bnf_test_data.shape)
log.debug(bnf_train_data.shape)


log.debug(bnf_valid_labels.shape)
log.debug(bnf_test_labels.shape)
log.debug(bnf_train_labels.shape)

bnf_valid_data_size = bnf_valid_data.shape[0] * percent
bnf_test_data_size = bnf_test_data.shape[0] * percent
bnf_train_data_size = bnf_train_data.shape[0] * percent


log.debug("size of data after taking percentage of original size")
log.debug(bnf_valid_data_size)
log.debug(bnf_test_data_size)
log.debug(bnf_train_data_size)

batch_size = 8192
bnf_train_gen =bcolz_data_generator(bnf_train_data, bnf_train_labels, batch_size=batch_size)
bnf_valid_gen =bcolz_data_generator(bnf_valid_data, bnf_valid_labels, batch_size=batch_size)
bnf_test_gen =bcolz_data_generator(bnf_test_data, bnf_test_labels, batch_size=batch_size)

## top model
import keras.backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Reshape
from keras import regularizers

alpha = 1
dropout=1e-3

if K.image_data_format() == 'channels_first':
    shape = (int(1024 * alpha), 1, 1)
else:
    shape = (1, 1, int(1024 * alpha))

classes = num_classes

top_model = Sequential()
top_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 1024)))
top_model.add(Reshape(shape, name='reshape_1'))
top_model.add(Dropout(dropout, name='dropout'))
top_model.add(Conv2D(classes, (1, 1),
           padding='same', name='conv_preds'))
top_model.add(Activation('softmax', name='act_softmax'))
top_model.add(Reshape((classes,), name='reshape_2'))

top_model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['accuracy'])
top_model.summary()

# fit the model
from keras.callbacks import EarlyStopping, ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.topmodel.mobilenet.hdf5', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
top_model.fit_generator(bnf_train_gen,
          steps_per_epoch= (1 + int(bnf_train_data_size // batch_size)),
          epochs=15,
          validation_data=bnf_valid_gen,
          validation_steps= (1 + int(bnf_valid_data_size // batch_size)),
          callbacks=[early_stopping, checkpointer])

y_true, y_pred = prediction_from_gen(gen=bnf_test_gen,
                                     steps=(1 + int(bnf_test_data_size // batch_size)),
                                     model=top_model,
                                     dirname=bnf_test_name)

# write to disk
test_result = 'bottleneck_features_mobilenet_result.npz'
log.debug(y_true.shape)
log.debug(y_pred.shape)
np.savez(test_result, y_true=y_true, y_pred=y_pred)

# read back and check
npzfile = np.load(test_result)
y_true = npzfile['y_true']
y_pred = npzfile['y_pred']
print(y_true.shape)
print(y_pred.shape)

# report test accuracy
test_accuracy = 100*np.sum(np.argmax(y_pred, axis=1)==np.argmax(y_true, axis=1))/len(y_true)
print('Test accuracy: %.4f%%' % test_accuracy)

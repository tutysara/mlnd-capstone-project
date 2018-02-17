import numpy as np
import pandas as pd
import bcolz
import time
from bcolzutils import *
from util import *
import logging
import datetime

import keras.backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Reshape
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras import optimizers
from keras.applications.mobilenet import MobileNet

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)


#percent = 1
percent = 0.005

basedir="/media/hdd/datastore/t4sa"
valid_name = basedir + '/valid_data'
test_name = basedir + '/test_data'
train_name = basedir +'/train_data'

bnf_valid_name = basedir + '/bottleneck_features_mobilenet_valid'
bnf_test_name = basedir + '/bottleneck_features_mobilenet_test'
bnf_train_name = basedir + '/bottleneck_features_mobilenet_train'

num_classes = 3
epochs = 15
batch_size = 32
test_prefix = ""
decay_rate = 0.1/5

if percent < 1:
    test_prefix = "_test"

top_model_weight_path = f'saved_models/weights.best.topmodel.mobilenet.hdf5'
full_model_weight_path = f'saved_models/weights.best.fullmodel.mobilenet{test_prefix}.hdf5'
test_result = f'finetune_fullmodel_mobilenet_result{test_prefix}.npz'
loss_history_csv_name = f'finetune_fullmodel_mobilenet_loss_history{test_prefix}.csv'

d = datetime.datetime.today()
log_filename=f"finetune_full_model_mobilenet_{d.year}-{d.month}-{d.day}-{d.hour}.{d.minute}.{d.second}{test_prefix}.log"

logging.basicConfig(level='DEBUG', filename= log_filename)
log = logging.getLogger(__name__)
log.debug("finetuning full model ")
log.debug("using top_model_weight_path" + top_model_weight_path)
log.debug("using full_model_weight_path" + full_model_weight_path)
log.debug("using test_result" + test_result)
log.debug("using loss_history_csv_name" + loss_history_csv_name)

## load original bcolz data from disk
# read from disk and check size
valid_data = bcolz.carray(rootdir= valid_name+'_data.bclz', mode='r')
test_data = bcolz.carray(rootdir= test_name + '_data.bclz', mode='r')
train_data = bcolz.carray(rootdir= train_name+ '_data.bclz', mode='r')


valid_labels = bcolz.carray(rootdir= valid_name+'_labels.bclz', mode='r')
test_labels = bcolz.carray(rootdir= test_name + '_labels.bclz', mode='r')
train_labels = bcolz.carray(rootdir= train_name+ '_labels.bclz', mode='r')

log.debug("loading original data from disk")
log.debug(valid_data.shape)
log.debug(test_data.shape)
log.debug(train_data.shape)

log.debug(valid_labels.shape)
log.debug(test_labels.shape)
log.debug(train_labels.shape)

valid_data_size = int(valid_data.shape[0]*percent)
test_data_size = int(test_data.shape[0]*percent)
train_data_size = int(train_data.shape[0]*percent)

if percent < 1:
    valid_data = valid_data[:valid_data_size]
    valid_labels = valid_labels[:valid_data_size]
    test_data = test_data[:test_data_size]
    test_labels = test_labels[:test_data_size]
    train_data = train_data[:train_data_size]
    train_labels = train_labels[:train_data_size]

log.debug("loading percentage of original data from disk")
log.debug(valid_data.shape)
log.debug(test_data.shape)
log.debug(train_data.shape)

log.debug(valid_labels.shape)
log.debug(test_labels.shape)
log.debug(train_labels.shape)

# top model

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

#top_model.compile(optimizer='adam',
#              loss='categorical_crossentropy', metrics=['accuracy'])
top_model.summary()

top_model.load_weights(top_model_weight_path)

# fine tune on full model
mobilenet_model = MobileNet(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# CREATE AN "REAL" MODEL FROM Mobilenet
# BY COPYING ALL THE LAYERS OF Mobilenet
model = Sequential()
for l in mobilenet_model.layers:
    model.add(l)


# CONCATENATE THE TWO MODELS
model.add(top_model)

# LOCK THE TOP CONV LAYERS
for layer in model.layers:
    layer.trainable = False

# COMPILE THE MODEL
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9, decay=decay_rate),
              metrics=['accuracy'])

model.summary()


# load train, test, and validation datasets
valid_data_gen = bcolz_data_generator(valid_data, valid_labels, batch_size=batch_size, progress=False)
test_data_gen = bcolz_data_generator(test_data, test_labels, batch_size=batch_size, progress=False)
train_data_gen = bcolz_data_generator(train_data, train_labels, batch_size=batch_size, progress=False)

checkpointer = ModelCheckpoint(filepath=full_model_weight_path, verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
csv_logger = CSVLogger(loss_history_csv_name, append=True, separator=',')

loss_history = model.fit_generator(train_data_gen,
          steps_per_epoch= (1+ train_data_size // batch_size),
          epochs=epochs,
          validation_data=valid_data_gen,
          validation_steps= (1+ valid_data_size // batch_size),
          callbacks=[early_stopping, checkpointer])

# loss_history
log.debug("loss_history\n"+ str(loss_history))
y_true, y_pred = prediction_from_gen(gen=test_data_gen,
                                     steps=(1 + test_data_size // batch_size),
                                     model=model,
                                     dirname=test_name)

# write to disk
log.debug(y_true.shape)
log.debug(y_pred.shape)
np.savez(test_result, y_true=y_true, y_pred=y_pred)

# read back and check
npzfile = np.load(test_result)
y_true = npzfile['y_true']
y_pred = npzfile['y_pred']
log.debug(y_true.shape)
log.debug(y_pred.shape)

# report test accuracy
test_accuracy = 100*np.sum(np.argmax(y_pred, axis=1)==np.argmax(y_true, axis=1))/len(y_true)
log.debug('Test accuracy: %.4f%%' % test_accuracy)

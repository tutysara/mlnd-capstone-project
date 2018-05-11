import numpy as np
import pandas as pd
import bcolz
import time
import logging
import datetime

import sys
sys.path.append('..')

from bcolzutils import *
from util import *

from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras import optimizers
from keras.regularizers import l2

from keras.applications.mobilenet import preprocess_input as mobile_preprocess_input

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

from vgg19_caffe_mode import get_model

arch = "caffe_vgg19"
basedir="/media/hdd/datastore/t4sa"
#basedir="/home/tutysara/src/myprojects/dog-project/dogImages"

#percent = 0.00099
percent = 1
#epochs=5
epochs=15
#num_classes = 133
num_classes = 3
batch_size = 48
#batch_size = 64
lr=1e-3
momentum=0.9
l2_weight_decay = 1e-5
test_prefix=""

def lr_schedule(epoch):
    """ divides the lr by 10 every 5 epochs"""
    n = (epoch + 1) // 5
    return lr / (10 ** n)

if percent < 1:
    test_prefix = "test_"

d = datetime.datetime.today()

model_path = f'saved_models/{test_prefix}all_layers_{arch}_weights.hdf5'
loss_history_csv_name = f'{test_prefix}all_layers.{arch}_loss_history.csv'
test_result = f'saved_models/{test_prefix}all_layers_{arch}_result.npz'
log_filename=f"all_layer_train_{arch}_{d.year}-{d.month}-{d.day}-{d.hour}.{d.minute}.{d.second}_{test_prefix}.log"


logging.basicConfig(level='DEBUG',
                    handlers=[logging.FileHandler(log_filename),
                              logging.StreamHandler()])
log = logging.getLogger(__name__)


log.debug("fine tune all layers")
log.debug("using all_model_weight_path :" + model_path)
log.debug("using test_result :" + test_result)
log.debug("using loss_history_csv_name :" + loss_history_csv_name)

train_name = basedir + '/pp_train_data'
valid_name = basedir + '/pp_valid_data'
test_name = basedir + '/pp_test_data'

temp_dir = "/tmp/"

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
## make a generator of loaded data

valid_data_gen = bcolz_data_generator(valid_data,
                                      valid_labels,
                                      batch_size=batch_size, progress=True, shuffle=True)
test_data_gen = bcolz_data_generator(test_data,
                                     test_labels,
                                     batch_size=batch_size, progress=True, shuffle=True)
train_data_gen = bcolz_data_generator(train_data,
                                      train_labels,
                                      batch_size=batch_size, progress=True, shuffle=True)

my_model = get_model(train_all_layer=True)
#my_model.summary()

for layer in my_model.layers:
    if hasattr(layer, 'kernel_regularizer'):
        layer.kernel_regularizer= regularizers.l2(l2_weight_decay)

for layer in my_model.layers:
    if hasattr(layer, 'kernel_regularizer'):
        log.debug("{}, {}, {}".format(layer.name,layer.trainable, layer.kernel_regularizer))
    else:
        log.debug("{},{}".format(layer.name,layer.trainable))


# fit the model
checkpointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
csv_logger = CSVLogger(loss_history_csv_name, append=True, separator=',')
lrscheduler = LearningRateScheduler(schedule=lr_schedule)

my_model.compile(loss='categorical_crossentropy',
          optimizer=optimizers.SGD(lr=lr, momentum=momentum),
          metrics=['accuracy'])

my_model.fit_generator(train_data_gen,
          steps_per_epoch= (1 + int(train_data_size // batch_size)),
          epochs=epochs,
          validation_data=valid_data_gen,
          validation_steps= (1 + int(valid_data_size // batch_size)),
          callbacks=[early_stopping, checkpointer, csv_logger, lrscheduler])

# load the best model
my_model.load_weights(model_path)

# calculate result
y_true, y_pred = prediction_from_gen(gen=test_data_gen,
                                     steps=(1 + int(train_data_size // batch_size)),
                                     model=my_model,
                                     dirname=temp_dir)

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

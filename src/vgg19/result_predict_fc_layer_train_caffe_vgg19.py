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

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras import optimizers
from keras import regularizers

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input

from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input as mobile_preprocess_input

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

## configs
arch = "caffe_vgg19"
basedir="/media/hdd/datastore/t4sa"
#basedir="/home/tutysara/src/myprojects/dog-project/dogImages"

#percent = 0.00099
percent = 1
epochs=5
#epochs=15
#num_classes = 133
num_classes = 3 
#batch_size = 48
batch_size = 64
lr=1e-3
momentum=0.9
l2_weight_decay = 1e-5
test_prefix=""

if percent < 1:
    test_prefix = "test_"

d = datetime.datetime.today()

model_path = f'saved_models/fc_layers_{arch}_weights.hdf5'
test_result = f'saved_models/{test_prefix}fc_layers_{arch}_result.npz'
log_filename=f"result_predict_fc_layers_train_{arch}_{d.year}-{d.month}-{d.day}-{d.hour}.{d.minute}.{d.second}_{test_prefix}.log"


logging.basicConfig(level='DEBUG',
                    handlers=[logging.FileHandler(log_filename),
                              logging.StreamHandler()])
log = logging.getLogger(__name__)


log.debug("prediction for fine tune fc layers")
log.debug("using fc_model_weight_path :" + model_path)
log.debug("using test_result :" + test_result)

train_name = basedir + '/pp_train_data'
valid_name = basedir + '/pp_valid_data'
test_name = basedir + '/pp_test_data'

temp_dir = "/tmp/" 

## data setup
# read bcolz data
bclz_valid_data = bcolz.carray(rootdir= valid_name+'_data.bclz', mode='r')
bclz_test_data = bcolz.carray(rootdir= test_name + '_data.bclz', mode='r')
bclz_train_data = bcolz.carray(rootdir= train_name+ '_data.bclz', mode='r')


bclz_valid_labels = bcolz.carray(rootdir= valid_name+'_labels.bclz', mode='r')
bclz_test_labels = bcolz.carray(rootdir= test_name + '_labels.bclz', mode='r')
bclz_train_labels = bcolz.carray(rootdir= train_name+ '_labels.bclz', mode='r')

print(bclz_valid_data.shape, bclz_valid_labels.shape, type(bclz_valid_data))
print(bclz_test_data.shape, bclz_test_labels.shape, type(bclz_test_data)) 
print(bclz_train_data.shape, bclz_train_labels.shape, type(bclz_train_data)) 

# take percentage of data if required
bclz_valid_data3 = bclz_valid_data
bclz_test_data3 = bclz_test_data
bclz_train_data3 = bclz_train_data

bclz_valid_labels3 = bclz_valid_labels
bclz_test_labels3 = bclz_test_labels
bclz_train_labels3 = bclz_train_labels

valid_len = int(len(bclz_valid_data) * percent)
test_len = int(len(bclz_test_data) * percent)
train_len = int(len(bclz_train_data) * percent)
    
if percent < 1:
    bclz_valid_data3 = bclz_valid_data[:valid_len]
    bclz_test_data3 = bclz_test_data[:test_len]
    bclz_train_data3 = bclz_train_data[:train_len]

    bclz_valid_labels3 = bclz_valid_labels[:valid_len]
    bclz_test_labels3 = bclz_test_labels[:test_len]
    bclz_train_labels3 = bclz_train_labels[:train_len]
    
print(bclz_valid_data3.shape, bclz_valid_labels3.shape, type(bclz_valid_data3))
print(bclz_test_data3.shape, bclz_test_labels3.shape, type(bclz_test_data3))
print(bclz_train_data3.shape, bclz_train_labels3.shape, type(bclz_train_data3))


test_gen =bcolz_data_generator(bclz_test_data3, bclz_test_labels3, batch_size=batch_size, shuffle=True)

## model
# Generate a model with all layers
vgg19 = VGG19(weights=None, include_top=True)

x = Dropout(0.5,name="dropout1")(vgg19.layers[-3].output)
x = vgg19.layers[-2](x)
x = Dropout(0.5,name="dropout2")(x)
x = Dense(num_classes, activation='softmax', name='my_predictions')(x)

my_model = Model(inputs=vgg19.input, outputs=x)
my_model.summary()

# load best weights
my_model.load_weights(model_path)

## calculate result
from timeit import default_timer as timer

# warmup
_, _ = prediction_from_gen(gen=test_gen,
                                     steps=3,
                                     model=my_model,
                                     dirname=temp_dir)
start = timer()
y_true, y_pred = prediction_from_gen(gen=test_gen,
                                     steps=(1 + int(train_len // batch_size)),
                                     model=my_model,
                                     dirname=temp_dir)
end = timer()
log.debug("Prediction time :"+ str(end - start))

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
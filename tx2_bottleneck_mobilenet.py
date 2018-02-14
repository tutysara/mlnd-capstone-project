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
log_filename=f"tx2_bottleneck_mobilenet_{d.year}-{d.month}-{d.day}-{d.hour}.{d.minute}.{d.second}.log"

logging.basicConfig(level='DEBUG', filename= log_filename)
log = logging.getLogger(__name__)
log.debug("transform data to bottleneck features")

basedir="/media/hdd/datastore/t4sa"
valid_name = basedir + '/valid_data'
test_name = basedir + '/test_data'
train_name = basedir +'/train_data'

bnf_valid_name = basedir + '/bottleneck_features_mobilenet_valid'
bnf_test_name = basedir + '/bottleneck_features_mobilenet_test'
bnf_train_name = basedir + '/bottleneck_features_mobilenet_train'

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

valid_data_size = valid_data.shape[0]*percent
test_data_size = test_data.shape[0]*percent
train_data_size = train_data.shape[0]*percent
"""
valid_data = valid_data[:int(valid_data_size)]
valid_labels = valid_labels[:int(valid_data_size)]
test_data = test_data[:int(test_data_size)]
test_labels = test_labels[:int(test_data_size)]
train_data = train_data[:int(train_data_size)]
train_labels = train_labels[:int(train_data_size)]
"""
log.debug("loading percentage of original data from disk")
log.debug(valid_data.shape)
log.debug(test_data.shape)
log.debug(train_data.shape)

log.debug(valid_labels.shape)
log.debug(test_labels.shape)
log.debug(train_labels.shape)
## make a generator of loaded data
batch_size = 256
valid_data_gen = bcolz_data_generator(valid_data,
                                      valid_labels,
                                      batch_size=batch_size, progress=True)
test_data_gen = bcolz_data_generator(test_data,
                                     test_labels,
                                     batch_size=batch_size, progress=True)
train_data_gen = bcolz_data_generator(train_data,
                                      train_labels,
                                      batch_size=batch_size, progress=True)

## Transform data and save it in bottleneck feacture format

from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input
import time

mobilenet_feature_ext = MobileNet(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
mobilenet_feature_ext._make_predict_function()

s= time.time()
remove_bcolz_dir(bnf_valid_name)
bnf_valid_data,  bnf_valid_labels = bcolz_prediction_writer(gen=valid_data_gen,
                        steps=(1+(valid_data.shape[0]//batch_size)),
                        model=mobilenet_feature_ext,
                        preprocess= mobilenet_preprocess_input,
                        dirname=bnf_valid_name)
log.debug("Took {:.2f} seconds to calculate bnf_valid_data".format(time.time()-s))

s= time.time()
remove_bcolz_dir(bnf_test_name)
bnf_test_data,  bnf_test_labels = bcolz_prediction_writer(gen=test_data_gen,
                        steps=(1+(test_data.shape[0]//batch_size)),
                        model=mobilenet_feature_ext,
                        preprocess= mobilenet_preprocess_input,
                        dirname=bnf_test_name)
log.debug("Took {:.2f} seconds to calculate bnf_test_data".format(time.time()-s))

s= time.time()
remove_bcolz_dir(bnf_train_name)
bnf_train_data,  bnf_train_labels = bcolz_prediction_writer(gen=train_data_gen,
                        steps=(1+(train_data.shape[0]//batch_size)),
                        model=mobilenet_feature_ext,
                        preprocess= mobilenet_preprocess_input,
                        dirname=bnf_train_name)
log.debug("Took {:.2f} seconds to calculate bottleneck_features_mobilenet_train".format(time.time()-s))

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


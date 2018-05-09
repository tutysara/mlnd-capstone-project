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

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

from keras.applications.vgg19 import VGG19
from keras.applications.imagenet_utils import preprocess_input

import time


arch = "vgg19"
batch_size = 16

percent = 0.005
percent = 1

d = datetime.datetime.today()
log_filename = f"tx2_bottleneck_{arch}_{d.year}-{d.month}-{d.day}-{d.hour}.{d.minute}.{d.second}.log"

logging.basicConfig(level='DEBUG',
                    handlers=[logging.FileHandler(log_filename),
                              logging.StreamHandler()])

log = logging.getLogger(__name__)
log.debug("transform data to bottleneck features")

#basedir="/media/hdd/datastore/t4sa"
basedir="/home/tutysara/src/myprojects/dog-project/dogImages"

valid_name = basedir + '/valid_data'
test_name = basedir + '/test_data'
train_name = basedir +'/train_data'

pp_valid_name = basedir + '/pp_valid_data'
pp_test_name = basedir + '/pp_test_data' 
pp_train_name = basedir + '/pp_train_data'

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
                                      batch_size=batch_size, progress=True,
                                      )
test_data_gen = bcolz_data_generator(test_data,
                                     test_labels,
                                     batch_size=batch_size, progress=True,
                                     )
train_data_gen = bcolz_data_generator(train_data,
                                      train_labels,
                                      batch_size=batch_size, progress=True,
                                      )
# identity model

class IdentityModel:
    def predict(self, x):
        return x
identity_model = IdentityModel()
    
## Transform data and save it in bottleneck feacture format
s= time.time()
remove_bcolz_dir(pp_valid_name)
pp_valid_data,  pp_valid_labels = bcolz_prediction_writer(gen=valid_data_gen,
                        steps=(1+(valid_data.shape[0]//batch_size)),
                        model=identity_model,
                        preprocess= preprocess_input,
                        dirname=pp_valid_name)
log.debug("Took {:.2f} seconds to calculate bnf_valid_data".format(time.time()-s))

s= time.time()
remove_bcolz_dir(pp_test_name)
pp_test_data,  pp_test_labels = bcolz_prediction_writer(gen=test_data_gen,
                        steps=(1+(test_data.shape[0]//batch_size)),
                        model=identity_model,
                        preprocess= preprocess_input,
                        dirname=pp_test_name)
log.debug("Took {:.2f} seconds to calculate bnf_test_data".format(time.time()-s))

s= time.time()
remove_bcolz_dir(pp_train_name)
pp_train_data,  pp_train_labels = bcolz_prediction_writer(gen=train_data_gen,
                        steps=(1+(train_data.shape[0]//batch_size)),
                        model=identity_model,
                        preprocess= preprocess_input,
                        dirname=pp_train_name)
log.debug("Took {:.2f} seconds to calculate bottleneck_features_mobilenet_train".format(time.time()-s))

## Read it back from disk and check size
pp_valid_data = bcolz.carray(rootdir= pp_valid_name+'_data.bclz', mode='r')
pp_test_data = bcolz.carray(rootdir= pp_test_name + '_data.bclz', mode='r')
pp_train_data = bcolz.carray(rootdir= pp_train_name+ '_data.bclz', mode='r')


pp_valid_labels = bcolz.carray(rootdir= pp_valid_name+'_labels.bclz', mode='r')
pp_test_labels = bcolz.carray(rootdir= pp_test_name + '_labels.bclz', mode='r')
pp_train_labels = bcolz.carray(rootdir= pp_train_name+ '_labels.bclz', mode='r')

log.debug("reading data from disk")
log.debug(pp_valid_data.shape)
log.debug(pp_test_data.shape)
log.debug(pp_train_data.shape)

log.debug(pp_valid_labels.shape)
log.debug(pp_test_labels.shape)
log.debug(pp_train_labels.shape)


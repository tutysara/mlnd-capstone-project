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

bnf_valid_name = basedir + 'bottleneck_features_mobilenet_valid'
bnf_test_name = basedir + 'bottleneck_features_mobilenet_test'
bnf_train_name = basedir + 'bottleneck_features_mobilenet_train'

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

bnf_valid_data = valid_data[:int(bnf_valid_data_size)]
bnf_valid_labels = valid_labels[:int(bnf_valid_data_size)]

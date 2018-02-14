import numpy as np
from sklearn.datasets import load_files
import pandas as pd
import bcolz
import time
from bcolzutils import *
from util import *



num_classes = 3
batch_size=256
#percent = 0.005
percent = 1

script_start_time = time.time()
# define function to load train, test, and validation datasets
basedir="/media/hdd/datastore/t4sa"
train_idx_path = basedir+ "/b-t4sa_train.txt"
valid_idx_path = basedir+ "/b-t4sa_val.txt"
test_idx_path = basedir+ "/b-t4sa_test.txt"

train_name = basedir + '/train_data'
valid_name = basedir + '/valid_data'
test_name = basedir + '/test_data'

col_names = ["X", "y"]
train_data_df = pd.read_csv(train_idx_path, sep=" ", header=None, names=col_names)
valid_data_df = pd.read_csv(valid_idx_path, sep=" ", header=None, names=col_names)
test_data_df = pd.read_csv(test_idx_path, sep=" ", header=None, names=col_names)

train_data_df = train_data_df[:int(train_data_df.shape[0]*percent)]
valid_data_df = valid_data_df[:int(valid_data_df.shape[0]*percent)]
test_data_df = test_data_df[:int(test_data_df.shape[0]*percent)]

print(train_data_df.shape)
print(valid_data_df.shape)
print(test_data_df.shape)

print("Total records =", train_data_df.shape[0] + valid_data_df.shape[0] + test_data_df.shape[0])




## code to get the data and save it as bcolz array, this will be read many times
train_gen = df_data_generator(train_data_df,
                              batch_size=batch_size,
                              transformer=img_to_tensor_transformer,
                              basedir=basedir,
                              num_classes=num_classes)

valid_gen = df_data_generator(valid_data_df,
                              batch_size=batch_size,
                              transformer=img_to_tensor_transformer,
                              basedir=basedir,
                              num_classes=num_classes)

test_gen = df_data_generator(test_data_df,
                             batch_size=batch_size,
                             transformer=img_to_tensor_transformer,
                             basedir=basedir,
                             num_classes=num_classes)
s= time.time()
remove_bcolz_dir(valid_name)
valid_data,  valid_labels = bcolz_writer(gen=valid_gen,
                        steps=(1+(valid_data_df.shape[0]//batch_size)),
                        dirname=valid_name)
print("Took {:.2f} seconds to calculate valid_data".format(time.time()-s))

s= time.time()
remove_bcolz_dir(test_name)
test_data,  test_labels = bcolz_writer(gen=test_gen,
                        steps=(1+(test_data_df.shape[0]//batch_size)),
                        dirname=test_name)
print("Took {:.2f} seconds to calculate test_data".format(time.time()-s))

s= time.time()
remove_bcolz_dir(train_name)
train_data,  train_labels = bcolz_writer(gen=train_gen,
                        steps=(1+(train_data_df.shape[0]//batch_size)),
                        dirname=train_name)
print("Took {:.2f} seconds to calculate train_data".format(time.time()-s))

print("percentage of data", percent)

print(valid_data.shape)
print(test_data.shape)
print(train_data.shape)

print(valid_labels.shape)
print(test_labels.shape)
print(train_labels.shape)

print("Took {:.2f} seconds to run script".format(time.time()-script_start_time))

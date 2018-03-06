import bcolz
import os
import time
import shutil
import logging

log = logging.getLogger(__name__)
def bcolz_writer(gen, steps, dirname, progress=False ):
    X = None; y = None
    data_dir = dirname + "_data.bclz"
    label_dir = dirname + "_labels.bclz"
    for i in range(steps):
        X_batch, y_batch = next(gen)
        if i== 0:
            X = bcolz.carray(X_batch, rootdir=data_dir, mode='w')
            y = bcolz.carray(y_batch, rootdir=label_dir, mode='w')
        else:
            X.append(X_batch)
            y.append(y_batch)
    X.flush()
    y.flush()
    return X, y

def remove_bcolz_dir(dirname):
    dirpath_data = os.path.abspath(dirname+ "_data.bclz")
    if os.path.isdir(dirpath_data):
        shutil.rmtree(dirpath_data)
    dirpath_labels = os.path.abspath(dirname+ "_labels.bclz")
    if os.path.isdir(dirpath_labels):
        shutil.rmtree(dirpath_labels)

def bcolz_prediction_writer(gen, steps, model, preprocess, dirname ):
    X = None; y = None
    data_dir = dirname + "_data.bclz"
    label_dir = dirname + "_labels.bclz"
    for i in range(steps):
        X_batch, y_batch = next(gen)
        if preprocess:
            X_out = model.predict(preprocess(X_batch))
        else:
            X_out = model.predict(X_batch)
        if i== 0:
            X = bcolz.carray(X_out, rootdir=data_dir, mode='w')
            y = bcolz.carray(y_batch, rootdir=label_dir, mode='w')
        else:
            X.append(X_out)
            y.append(y_batch)
    X.flush()
    y.flush()
    return X, y

def bcolz_data_generator(bclz_data, bclz_labels, batch_size=32, progress=False, preprocess=None):
    while True:
        max_range = (1 + bclz_data.shape[0]//batch_size)
        for i in range(max_range):
            s = time.time()
            curr_batch_X, curr_batch_y = bclz_data[i*batch_size : (i+1)*batch_size], bclz_labels[i*batch_size : (i+1)*batch_size]
            X_out = curr_batch_X
            if preprocess:
                X_out = preprocess(curr_batch_X)                
            yield (X_out, curr_batch_y)
            if progress and max_range>1:
                log.debug("bcolz.gen Iteration {i}/{t} took {s:.2f}s".format(i=i, t=max_range, s=(time.time()-s)))

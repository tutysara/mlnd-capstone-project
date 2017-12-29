# convert and load images
from keras.preprocessing import image
from keras.utils import np_utils
from tqdm import tqdm
from PIL import ImageFile
import time
import numpy as np
import bcolz
ImageFile.LOAD_TRUNCATED_IMAGES = True

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in img_paths]
    return np.vstack(list_of_tensors)


#definition of data generator
def df_data_generator(df, batch_size=32, num_classes=3, shuffle=False, basedir=".", transformer=None, progress=True):
    while True:
        if shuffle:
            df = df.sample(frac=1)

        X_file_name = df.X.apply(lambda x: basedir+"/"+x)
        y = np_utils.to_categorical(df.y, num_classes)
    # infinitely serve batches
        max_range = (1+ df.shape[0]//batch_size)
        for i in range(max_range):
            s = time.time()
            if transformer:
                yield transformer(X_file_name[i*batch_size : (i+1)*batch_size]).astype('float32'), y[i*batch_size : (i+1)*batch_size]
            else:
                yield X_file_name[i*batch_size : (i+1)*batch_size], y[i*batch_size : (i+1)*batch_size]
            if progress and max_range>1:
                print("df.gen Iteration {i}/{t} took {s:.2f}s".format(i=i, t=max_range, s=(time.time()-s)))

def img_to_tensor_transformer(x):
    return paths_to_tensor(x)

def prediction_from_gen(gen, steps, model, dirname ):
    y_true = None; y_pred = None
    yt_dir = dirname + "_y_true.bclz"
    yp_dir = dirname + "_y_pred.bclz"
    for i in range(steps):
        X_batch, y_batch = next(gen)
        y_out = model.predict(X_batch)
        if i== 0:
            y_true = bcolz.carray(y_batch, rootdir=yt_dir, mode='w')
            y_pred = bcolz.carray(y_out, rootdir=yp_dir, mode='w') 
        else:
            y_true.append(y_batch)
            y_pred.append(y_out)
    y_true.flush()
    y_pred.flush()
    return y_true, y_pred

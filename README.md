## Python setup
Install python 3 and use the requirements.txt to install dependencies

## Data setup
Download the data from -- http://www.t4sa.it/
The scripts expect the data to be present in -- /media/hdd/datastore/t4sa, create the data directory if it doesn't exist

```
mkdir -p /media/hdd/datastore/t4sa
```

unzip the contents of the tar file inside the data directory.


## One time processing
The data has to be convered to bcolz format.
This is done by running the preprocessing scripts

```
ipython src/load_data_to_bcolz.py
ipython src/pp_t4sa_data.py
```

## Training and testing the model
Each model has a training script and it should be run to rain the models. It is of the form
{fc|all}_layer_train_<model>.py.

For example in case of VGG19 model we have
```
all_layer_train_caffe_vgg19.py
fc_layer_train_caffe_vgg19.py
```

they can be run to train the model

```
ipython all_layer_train_caffe_vgg19.py
ipython fc_layer_train_caffe_vgg19.py
```

We also have scripts to run the model on test set and save the result

For example in case of VGG19, the model can be tested and the results are saved by running

```
ipython result_predict_all_layer_train_caffe_vgg19.py
ipython result_predict_fc_layer_train_caffe_vgg19.py
```




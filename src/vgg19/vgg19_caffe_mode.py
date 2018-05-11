import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input

from keras.applications.mobilenet import MobileNet

def get_model(num_classes=3, alpha=1.0, dropout=1e-3, train_all_layer=False):
    ## top model
    # Generate a model with all layers (with top)
    vgg19 = VGG19(weights='imagenet', include_top=True)

    x = Dropout(0.5,name="dropout1")(vgg19.layers[-3].output)
    x = vgg19.layers[-2](x)
    x = Dropout(0.5,name="dropout2")(x)
    x = Dense(num_classes, activation='softmax', name='my_predictions')(x)

    # unfreeze all layers
    for layer in vgg19.layers:
        layer.trainable = train_all_layer

    my_model = Model(inputs=vgg19.input, outputs=x)

    #Then create the corresponding model
    my_model.layers[-5].trainable = True
    my_model.layers[-3].trainable = True
    my_model.layers[-1].trainable = True
    return my_model

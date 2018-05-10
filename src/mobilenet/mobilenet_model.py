import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Reshape, Conv2D, Activation
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras import optimizers
from keras.regularizers import l2 

from keras.applications.mobilenet import MobileNet

def get_model(num_classes=3, alpha=1.0, dropout=1e-3, train_all_layer=False):
    ## top model
    mobilenet = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling=None)
    if K.image_data_format() == 'channels_first':
        shape = (int(1024 * alpha), 1, 1)
    else:
        shape = (1, 1, int(1024 * alpha))

    for layer in mobilenet.layers:
        layer.trainable = train_all_layer


    ## from keras source for mobilenet  
    x = mobilenet.output
    x = GlobalAveragePooling2D()(x)
    x = Reshape(shape, name='reshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(256, (1, 1),
            padding='same', name='conv_preds')(x)
    x = Dense(128, name='dense11')(x)
    x = Dropout(dropout, name='dropout12')(x)

    x = Dense(num_classes, name='dense2')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((num_classes,), name='reshape_2')(x)

    my_model = Model(inputs=mobilenet.input, outputs=x)
    return my_model

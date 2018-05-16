import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.applications.inception_resnet_v2 import InceptionResNetV2

def get_model(num_classes=3, alpha=1.0, dropout=1e-3, train_all_layer=False):
## top model
    inceptionresnetv2 = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling=None)

    classes = num_classes
    for layer in inceptionresnetv2.layers:
        layer.trainable = False


    ## from keras source for InceptionResNetV2
    x = inceptionresnetv2.output
    #set_trace()
    print(x)
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    """
    x = Dropout(dropout, name='dropout_top1')(x)
    x = Dense(512, name='dense_top1')(x)
    x = Dropout(dropout, name='dropout_top2')(x)
    x = Dense(256, name='dense_top2')(x)
    x = Dropout(dropout, name='dropout_top3')(x)
    """
    x = Dense(classes, activation='softmax', name='predictions')(x)

    my_model = Model(inputs=inceptionresnetv2.input, outputs=x)
    return my_model

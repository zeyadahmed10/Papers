import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.callbacks import *

class MobileNetV1:
    def __init__(self,weights_path=None):
        self.weights_path=weights_path


    hyper_prams=list()
    hyper_prams.append(['same', (1, 1), 64])
    hyper_prams.append(['valid', (2, 2), 128])
    hyper_prams.append(['same', (1, 1), 128])
    hyper_prams.append(['valid', (2, 2), 256])
    hyper_prams.append(['same', (1, 1), 256])
    hyper_prams.append(['valid', (2, 2), 512])
    hyper_prams.append(['same', (1, 1), 512])
    hyper_prams.append(['same', (1, 1), 512])
    hyper_prams.append(['same', (1, 1), 512])
    hyper_prams.append(['same', (1, 1), 512])
    hyper_prams.append(['same', (1, 1), 512])
    hyper_prams.append(['valid', (2, 2), 1024])
    hyper_prams.append(['same', (1, 1), 1024])


    def get_dw_sep_block(self, tensor, padtype, strides, chan, zero_padding_layer):
        x = tensor
        if zero_padding_layer == True:
            x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
        # Depthwise
        x = DepthwiseConv2D(kernel_size=(3, 3),
                            strides=strides,
                            use_bias=False,
                            padding=padtype)(x)
        x = BatchNormalization()(x)
        x = ReLU(max_value=6.0)(x)

        # Pointwise
        x = Conv2D(chan,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   use_bias=False,
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU(max_value=6.0)(x)
        return x

    def create_mobile_net(self):
        tensor = Input((224, 224, 3), name="input_2")
        hidden = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_last',
                        dilation_rate=(1, 1), use_bias=False, kernel_initializer='GlorotUniform', name='conv1')(tensor)
        hidden = BatchNormalization(axis=3, name='conv1_bn')(hidden)
        hidden = ReLU(max_value=6.0, name='conv1_relu')(hidden)
        for i in range(13):
            flag = False
            if i in [1, 3, 5, 11]:
                flag = True
            hidden = self.get_dw_sep_block(hidden, self.hyper_prams[i][0], self.hyper_prams[i][1], self.hyper_prams[i][2], flag)
        hidden = GlobalAveragePooling2D()(hidden)
        hidden = Reshape(target_shape=(1, 1, 1024))(hidden)
        hidden = Dropout(0.001)(hidden)
        hidden = Conv2D(1000, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True)(hidden)
        hidden = Reshape(target_shape=(1000,))(hidden)
        output = Activation('softmax')(hidden)
        model = Model(inputs=tensor, outputs=output)
        if self.weights_path is not None:
            model.load_weights(self.weights_path)
        return model

## Creating an object from the MobileNet version 1
##pass weight path if there is
MobileNet= MobileNetV1()
##Building the Model
model= MobileNet.create_mobile_net()
model.summary()


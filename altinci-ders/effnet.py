from keras.layers import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers import DepthwiseConv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Input
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.models import Model

def get_top(x_input):
    x = LeakyReLU()(x_input)
    x = BatchNormalization()(x)
    return x

def get_block(x_input, input_channels, output_channels):
    x = Conv2D(input_channels, kernel_size=(1, 1), padding='same', use_bias=False)(x_input)
    x = get_top(x)
    # depthwise convolution işlemi her bir kanalda ayrı ayrı çalışarak hesaplama maliyetini azaltır
    x = DepthwiseConv2D(kernel_size=(1, 3), padding='same', use_bias=False)(x)
    x = get_top(x)
    x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(x)
    x = DepthwiseConv2D(kernel_size=(3, 1), padding='same', use_bias=False)(x)
    x = get_top(x)
    x = Conv2D(output_channels, kernel_size=(2, 1), strides=(1, 2), padding='same', use_bias=False)(x)
    return x


def EffNet(input_shape, num_classes):
    x_input = Input(shape=input_shape)
    x = get_block(x_input, 32, 64)
    x = get_block(x, 64, 128)
    x = get_block(x, 128, 256)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=x_input, outputs=x)
    return model


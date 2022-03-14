
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Dense
from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Model

def se_layer(x, num_filters, reduction=16):
    x_init = x

    x = GlobalAveragePooling2D()(x)
    x = Dense(num_filters//reduction, use_bias=False, activation="relu")(x)
    x = Dense(num_filters, use_bias=False, activation="sigmoid")(x)
    x = x * x_init
    return x

def residual_block(x, num_filters):
    x_init = x

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)

    s = Conv2D(num_filters, 1, padding="same")(x_init)
    s = BatchNormalization()(s)
    s = se_layer(s, num_filters)

    x = Activation("relu")(x + s)
    return x

def strided_conv_block(x, num_filters):
    x = Conv2D(num_filters, 3, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def encoder_block(x, num_filters):
    x1 = residual_block(x, num_filters)
    x2 = strided_conv_block(x1, num_filters)
    x3 = residual_block(x2, num_filters)
    p = MaxPool2D((2, 2))(x3)

    return x1, x3, p

def build_colonsegnet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Encoder """
    s11, s12, p1 = encoder_block(inputs, 64)
    s21, s22, p2 = encoder_block(p1, 256)

    """ Decoder 1 """
    x = Conv2DTranspose(128, 4, strides=4, padding="same")(s22)
    x = Concatenate()([x, s12])
    x = residual_block(x, 128)
    r1 = x

    x = Conv2DTranspose(128, 4, strides=2, padding="same")(s21)
    x = Concatenate()([x, r1])
    x = residual_block(x, 128)

    """ Decoder 2 """
    x = Conv2DTranspose(64, 4, strides=2, padding="same")(x)
    x = Concatenate()([x, s11])
    x = residual_block(x, 64)
    r2 = x

    x = Conv2DTranspose(32, 4, strides=2, padding="same")(s12)
    x = Concatenate()([x, r2])
    x = residual_block(x, 32)

    """ Output """
    output = Conv2D(1, 1, padding="same")(x)

    """ Model """
    model = Model(inputs, output)

    return model

if __name__ == "__main__":
    input_shape = (512, 512, 3)
    model = build_colonsegnet(input_shape)
    model.summary()

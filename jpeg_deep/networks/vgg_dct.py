from typing import Tuple

from keras import backend as K
from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Conv2DTranspose, Concatenate, GlobalAveragePooling2D
from keras.regularizers import l2


def VGG16_dct(classes: int=1000):
    """Instantiates the VGG16 DCT architecture.

    # Argument:
        - classes: The number of classes the network should predict.

    # Returns:
        A Keras model instance.
    """
    input_shape_y = (28, 28, 64)
    input_shape_cbcr = (14, 14, 128)

    input_y = Input(input_shape_y)
    input_cbcr = Input(input_shape_cbcr)

    norm_cbcr = BatchNormalization(
        name="b_norm_128", input_shape=input_shape_cbcr)(input_cbcr)

    # Block 1
    x = BatchNormalization(
        name="b_norm_64", input_shape=input_shape_y)(input_y)

    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block1_conv1_dct_256')(x)

    # Block 4
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv1_dct')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    concat = Concatenate(axis=-1)([x, norm_cbcr])

    # Block 5
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv1_dct')(concat)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu',
              kernel_regularizer=l2(0.0005), name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu',
              kernel_regularizer=l2(0.0005), name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    return Model(inputs=[input_y, input_cbcr], outputs=x)


def VGG16_dct_conv(classes:int=1000, input_shape: Tuple[int]=None):
    """ This is a modified version of the VGG16 DCT network to be fully convolutional.

    # Arguments:
        - classes: The number of classes to predict.
        - input_shape: The dimension of the inputs (x, y).
        
    # Returns:
        A Keras model instance.
    """
    if input_shape is None:
        input_shape_y = (None, None, 64)
        input_shape_cbcr = (None, None, 128)
    else:
        input_shape_y = (*input_shape, 64)
        input_shape_cbcr = (input_shape[0] // 2, input_shape[1] // 2, 128)

    input_y = Input(input_shape_y)
    input_cbcr = Input(input_shape_cbcr)

    norm_cbcr = BatchNormalization(
        name="b_norm_128", input_shape=input_shape_cbcr)(input_cbcr)

    # Block 1
    x = BatchNormalization(
        name="b_norm_64", input_shape=input_shape_y)(input_y)

    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block1_conv1_dct_256')(x)

    # Block 4
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv1_dct')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    concat = Concatenate(axis=-1)([x, norm_cbcr])

    # Block 5
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv1_dct')(concat)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Conv2D(4096, (7, 7), activation='relu',
               name='conv2d_1')(x)
    x = Conv2D(4096, (1, 1), activation='relu',
               name='conv2d_2')(x)
    x = Conv2D(classes, (1, 1), activation='softmax',
               name='conv2d_3')(x)
    x = GlobalAveragePooling2D()(x)

    return Model(inputs=[input_y, input_cbcr], outputs=x)


def VGG16_dct_deconv(classes:int=1000, input_shape:Tuple[int]=(28, 28)):
    """Instantiates the VGG16 DCT architecture.

    # Argument:
        - classes: The number of classes the network should predict.

    # Returns:
        A Keras model instance.
    """
    input_shape_y = (*input_shape, 64)
    input_shape_cb = (input_shape[0] // 2, input_shape[1] // 2, 64)
    input_shape_cr = (input_shape[0] // 2, input_shape[1] // 2, 64)

    input_y = Input(input_shape_y)
    input_cb = Input(shape=input_shape_cb)
    input_cr = Input(shape=input_shape_cr)

    cb = Conv2DTranspose(64, kernel_size=(2, 2), strides=2,
                         kernel_regularizer=l2(0.0005), name="deconv_cb")(input_cb)
    cr = Conv2DTranspose(64, kernel_size=(2, 2), strides=2,
                         kernel_regularizer=l2(0.0005), name="deconv_cr")(input_cr)

    x = Concatenate(axis=-1)([input_y, cb, cr])

    x = BatchNormalization(
        name="b_norm", input_shape=input_shape_y)(x)

    # Block 4
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv1_dct')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv1_dct')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu',
              kernel_regularizer=l2(0.0005), name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu',
              kernel_regularizer=l2(0.0005), name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    return Model(inputs=[input_y, input_cb, input_cr], outputs=x)


def VGG16_dct_deconv_conv(classes:int=1000, input_shape:Tuple[int]=None):
    """ This is a modified version of the VGG16 DCT network to be fully convolutional.

    # Arguments:
        - classes: The number of classes to predict.
        - input_shape: The dimension of the inputs (x, y).
        
    # Returns:
        A Keras model instance.
    """
    if input_shape is None:
        input_shape_y = (None, None, 64)
        input_shape_cb = (None, None, 64)
        input_shape_cr = (None, None, 64)
    else:
        input_shape_y = (*input_shape, 64)
        input_shape_cb = (input_shape[0] // 2, input_shape[1] // 2, 64)
        input_shape_cr = (input_shape[0] // 2, input_shape[1] // 2, 64)

    input_y = Input(input_shape_y)
    input_cb = Input(shape=input_shape_cb)
    input_cr = Input(shape=input_shape_cr)

    cb = Conv2DTranspose(64, kernel_size=(2, 2), strides=2,
                         kernel_regularizer=l2(0.0005), name="deconv_cb")(input_cb)
    cr = Conv2DTranspose(64, kernel_size=(2, 2), strides=2,
                         kernel_regularizer=l2(0.0005), name="deconv_cr")(input_cr)

    x = Concatenate(axis=-1)([input_y, cb, cr])

    x = BatchNormalization(
        name="b_norm", input_shape=input_shape_y)(x)

    # Block 4
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv1_dct')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv1_dct')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Conv2D(4096, (7, 7), activation='relu', name='conv2d_1')(x)
    x = Conv2D(4096, (1, 1), activation='relu', name='conv2d_2')(x)
    x = Conv2D(classes, (1, 1), activation='softmax', name='conv2d_3')(x)
    x = GlobalAveragePooling2D()(x)

    return Model(inputs=[input_y, input_cb, input_cr], outputs=x)


def VGG16_dct_y(classes:int=1000):
    """Instantiates the VGG16 DCT architecture.

    # Argument:
        - classes: The number of classes the network should predict.

    # Returns:
        A Keras model instance.
    """
    input_shape_y = (28, 28, 64)

    input_y = Input(input_shape_y)

    # Block 1
    x = BatchNormalization(
        name="b_norm_64", input_shape=input_shape_y)(input_y)

    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block1_conv1_dct_256')(x)

    # Block 4
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv1_dct')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv1_dct')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu',
              kernel_regularizer=l2(0.0005), name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu',
              kernel_regularizer=l2(0.0005), name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    return Model(input_y, outputs=x)


def VGG16_dct_y_conv(classes:int=1000, input_shape:Tuple[int]=(None, None)):
    """ This is a modified version of the VGG16 DCT network to be fully convolutional.

    # Arguments:
        - classes: The number of classes to predict.
        - input_shape: The dimension of the inputs (x, y). Can be (None, None).
        
    # Returns:
        A Keras model instance.
    """
    input_shape_y = (*input_shape, 64)

    input_y = Input(input_shape_y)

    # Block 1
    x = BatchNormalization(
        name="b_norm_64", input_shape=input_shape_y)(input_y)

    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block1_conv1_dct_256')(x)

    # Block 4
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv1_dct')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv1_dct')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block

    x = Conv2D(4096, (7, 7), activation='relu', name='conv2d_1')(x)
    x = Conv2D(4096, (1, 1), activation='relu', name='conv2d_2')(x)
    x = Conv2D(classes, (1, 1), activation='softmax', name='conv2d_3')(x)
    x = GlobalAveragePooling2D()(x)

    return Model(input_y, outputs=x)

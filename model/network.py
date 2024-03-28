"""
    基干网络
"""
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, Input, Convolution2D, \
    GlobalAveragePooling2D, Reshape, Permute
from keras import backend as K, Model
# from numpy import multiply
from keras.layers import multiply


# 原始网络
def build_base_network(input_shape):
    seq = Sequential()

    nb_filter = [6, 12]
    kernel_size = (3, 3)  # 使用元组定义kernel_size

    # convolutional layer 1
    seq.add(Conv2D(nb_filter[0], kernel_size, input_shape=input_shape, padding='valid', data_format='channels_first'))

    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    seq.add(Dropout(0.25))

    # convolutional layer 2
    seq.add(Conv2D(nb_filter[1], kernel_size, padding='valid', data_format='channels_first'))

    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    seq.add(Dropout(0.25))

    # flatten
    seq.add(Flatten())
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(50, activation='relu'))
    return seq


"""
    注意力机制区域
"""


def squeeze_excite_block(input, ratio=16):
    init = input
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def build_base_network_with_senet(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(6, (3, 3), padding='valid', data_format='channels_first')(inputs)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Add SE block
    x = squeeze_excite_block(x)

    x = Conv2D(12, (3, 3), padding='valid', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Add another SE block
    x = squeeze_excite_block(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation='relu')(x)

    model = Model(inputs=inputs, outputs=x)
    return model

"""
    基干网络
"""
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, Input, Convolution2D


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


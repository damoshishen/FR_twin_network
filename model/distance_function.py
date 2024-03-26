"""
    将距离函数keras层化，这样的好处是可以把该死的Lambda函数去掉
"""
from keras.layers import Layer
# import keras.backend as K
from keras import backend as K


# 欧氏距离
class EuclideanDistanceLayer(Layer):
    def __init__(self, **kwargs):
        super(EuclideanDistanceLayer, self).__init__(**kwargs)

    def call(self, inputs):
        x, y = inputs
        return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

    def compute_output_shape(self, input_shape):
        shape1, shape2 = input_shape
        print(shape1[0])
        return (shape1[0], 1)

"""
    将距离函数keras层化，这样的好处是可以把该死的Lambda函数去掉
    修正模型保存问题--2024.4.2
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


# 马氏距离
class MahalanobisDistanceLayer(Layer):
    def __init__(self, inverse_covariance, **kwargs):
        self.inverse_covariance = K.constant(inverse_covariance)  # 将逆协方差矩阵转换为Keras张量
        super(MahalanobisDistanceLayer, self).__init__(**kwargs)

    def call(self, inputs):
        x, y = inputs
        # diff = K.expand_dims(x - y, -1)  # 将差值变形以适应矩阵乘法
        diff = x - y
        temp = K.dot(diff, self.inverse_covariance)  # 计算 (x-y) 和 逆协方差矩阵的乘积
        # mahalanobis_square = K.dot(K.permute_dimensions(temp, (0, 2, 1)), diff)  # 计算马氏距离平方
        mahalanobis_square = K.sum(temp * diff, axis=1, keepdims=True)
        return K.sqrt(K.maximum(mahalanobis_square, K.epsilon()))  # 计算马氏距离并确保数值稳定性

    def compute_output_shape(self, input_shape):
        shape1, _ = input_shape
        return (shape1[0], 1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'inverse_covariance': K.eval(self.inverse_covariance).tolist()
            }
        )
        return config
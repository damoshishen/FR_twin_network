import numpy as np

from model.distance_function import *
import cv2
import tensorflow as tf
from keras.models import load_model


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def compute_accuracy(predictions, labels):
    predictions = tf.reshape(predictions, [-1])  # 将预测结果展平
    a = tf.reduce_mean(tf.cast(labels[predictions < 0.5], dtype=tf.float32))
    return tf.where(a > 1, tf.ones_like(a), a)


# 加载模型
model = load_model('../model/h5/test.h5', custom_objects={'MahalanobisDistanceLayer': MahalanobisDistanceLayer,
                                                         'contrastive_loss': contrastive_loss,
                                                         'compute_accuracy': compute_accuracy})

# 读取图像
img_1 = cv2.imread('../data/s1/1.pgm')
img_2 = cv2.imread('../data/s1/10.pgm')

# img_1 = cv2.resize(img_1, (56, 46, 3))
# img_2 = cv2.resize(img_2, (56, 46, 3))

# print(img_1.shape, img_2.shape)

img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
#
img_1 = cv2.resize(img_1, (46, 56))
img_2 = cv2.resize(img_2, (46, 56))
#
# # img_1 = np.moveaxis(img_1, -1, 1)
# # img_2 = np.moveaxis(img_2, -1, 1)
#
img_1 = np.expand_dims(img_1, axis=0)
img_2 = np.expand_dims(img_2, axis=0)
#
# print(img_1.shape, img_2.shape)
#
a = np.expand_dims(img_1, axis=0) / 255
b = np.expand_dims(img_2, axis=0) / 255
c = model.predict([a, b])
#
print(c)
print(c.ravel() < 0.5)
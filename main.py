"""
    测试人脸识别文件
        --简易版，基于opencv
"""
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
model = load_model('./model/h5/test.h5', custom_objects={'MahalanobisDistanceLayer': MahalanobisDistanceLayer,
                                                         'contrastive_loss': contrastive_loss,
                                                         'compute_accuracy': compute_accuracy})
test_pic = cv2.imread('./model/picture/processed_image.jpg')
test_pic = cv2.cvtColor(test_pic, cv2.COLOR_BGR2GRAY)
test_pic = cv2.resize(test_pic, (56, 46))
test_pic = np.expand_dims(test_pic, axis=0)
test_pic = np.moveaxis(test_pic, -1, 1)


saved = False

# 加载Haar级联人脸检测模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 打开默认摄像头
cap = cv2.VideoCapture(0)

while True:
    # 逐帧捕获
    ret, frame = cap.read()
    if not ret:
        break

    # 将捕获的帧转换为灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # (480, 640)

    # 在灰度帧上进行人脸检测
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 为每个检测到的人脸画矩形框
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 裁剪
        cropped_face = gray[y:y+h, x:x+w]

        # 转换形状
        processed_frame = cv2.resize(cropped_face, (56, 46))
        # print(processed_frame.shape)

        # 测试截取的图片
        if not saved:
            # 保存处理后的图像到本地
            cv2.imwrite('./model/picture/processed_image.jpg', processed_frame)
            saved = True

        # processed_frame = processed_frame / 255.0
        processed_frame = np.expand_dims(processed_frame, axis=0)
        processed_frame = np.moveaxis(processed_frame, -1, 1)
        print(processed_frame.shape, test_pic.shape)

        a = np.expand_dims(processed_frame, axis=0) / 255
        b = np.expand_dims(test_pic, axis=0) / 255
        c = model.predict([a, b])

        print(c)
        print(c.ravel() < 0.5)

    # 显示结果帧
    cv2.imshow('frame', frame)

    # 按'q'退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()

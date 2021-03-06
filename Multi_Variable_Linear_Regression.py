# Multi Variable Linear Regression

import tensorflow as tf
import numpy as np

tf.enable_eager_execution()
tf.__version__

# 1개의 bias와 여러 개의 변수
x1 = [ 73.,  93.,  89.,  96.,  73.]
x2 = [ 80.,  88.,  91.,  98.,  66.]
x3 = [ 75.,  93.,  90., 100.,  70.]
Y  = [152., 185., 180., 196., 142.]

# 가중치 설정
w1 = tf.Variable(10.)
w2 = tf.Variable(10.)
w3 = tf.Variable(10.)
b  = tf.Variable(10.)

learning_rate = 0.000001

for i in range(1000+1):
    # 비용을 기록하기 위한 GradientTape() 함수
    with tf.GradientTape() as tape:
        hypothesis = w1 * x1 +  w2 * x2 + w3 * x3 + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
    # 기울기 값 갱신
    w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1, w2, w3, b])
    
    # 가중치와 bias 갱신
    w1.assign_sub(learning_rate * w1_grad)
    w2.assign_sub(learning_rate * w2_grad)
    w3.assign_sub(learning_rate * w3_grad)
    b.assign_sub(learning_rate * b_grad)
    
    if i % 50 == 0:
      print("{:5} | {:12.4f}".format(i, cost.numpy()))

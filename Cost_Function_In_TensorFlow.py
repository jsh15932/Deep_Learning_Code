import tensorflow as tf
import numpy as np

X = np.array([1,2,3])
Y = np.array([1,2,3])

def cost_func(W, X, Y):
    hypothesis = X * W
    return tf.reduce_mean(tf.square(hypothesis - Y))
    """가설의 y값에서 y데이터를 뺀 값의 제곱의 평균이 비용이 됨"""

W_values = np.linspace(-3, 5, num=15)
cost_values = []

# W 값의 변화에 따른 비용의 결과를 출력
for feed_W in W_values:
    curr_cost = cost_func(feed_W, X, Y)
    cost_values.append(curr_cost)
    print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))

# Linear Regression - Gradient Descent
# Gradient Descent는 cost의 값을 최소화시키는 W, b의 값을 찾아가는 알고리즘입니다.

import tensorflow as tf

tf.enable_eager_execution()

# data 세팅
x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]

# 기울기 W 와 bias 세팅
W = tf.Variable(2.9)
b = tf.Variable(0.5)

learning_rate = 0.01

# Gradient Descent
for i in range(1000):
    # 변수들의 정보를 tape 에 기록
    with tf.GradientTape() as tape:
        hypothesis = W + x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
        """cost는 hypothesis의 y값에서 y_data의 값을 뺀 결과의 제곱 형태"""
    # W 기울기, b 기울기를 각각 할당
    W_grad, b_grad = tape.gradient(cost, [W, b])
    # W, b 값 갱신
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    """A.assign_sub(B) : A -= B"""
    # W, b 값 변화 출력
    if i % 100 == 0:
        print("{:5}|{:10.4f}|{:10.4}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))

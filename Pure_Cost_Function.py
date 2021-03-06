# Pure Cost Function

import numpy as np

X = np.array([1,2,3])
Y = np.array([1,2,3])

# 비용 함수 정의
def cost_func(W, X, Y):
    c = 0
    for i in range(len(X)):
        c += (W * X[i] - Y[i]) ** 2
        """X의 길이만큼, hypothesis의 y값에서 y데이터를 뺀 값을 비용에 더함""" 
    return c / len(X) """비용의 평균"""

# feed_W 값에 따라 변하는 비용의 값을 출력
for feed_W in np.linspace(-3, 5, num=15):
    curr_cost = cost_func(feed_W, X, Y)
    print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))

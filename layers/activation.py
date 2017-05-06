# Created by wz on 17-3-23.
# encoding=utf-8
import numpy as np


class ReLU:
    def __init__(self):
        self.type = 'relu'

    def forward(self, x):
        self.x = x
        self.y = np.select([x > 0], [x], 0)
        return self.y

    def backward(self, d):
        return np.select([self.x > 0], [d], 0)


class Sigmoid:
    def __init__(self):
        self.type = 'sigmoid'

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def forward(self, x):
        self.x = x
        self.y = self.sigmoid(x)
        return self.y

    def backward(self, d):
        self.dx = d * self.y * (1 - self.y)
        return self.dx


class TanH:
    def __init__(self):
        self.type = 'tanh'

    def forward(self, x):
        self.x = x
        a = np.exp(x)
        b = np.exp(-x)
        self.y = (a - b) / (a + b)
        return self.y

    def backward(self, d):
        self.dx = d * (1 - np.square(self.y))
        return self.dx


def test():
    relu = ReLU()
    print relu.forward(np.array([[1, -3], [100, -1000]]))
    print relu.backward(np.array([[1, -3], [100, -1000]]))


if __name__ == '__main__':
    test()

# Created by wz on 17-3-23.
# encoding=utf-8
import numpy as np


class FullyConnect:
    def __init__(self, l_in, l_out, relu=False):
        self.type = 'fullyconnect'
        self.weights = np.random.randn(l_out, l_in)
        self.weights /= np.sqrt(l_in)  # Xavier and ReLU(KaiMing He) paper
        if relu:
            self.weights *= np.sqrt(2)
        self.bias = np.random.randn(l_out, 1)
        self.lr = 0  # 参数由solver统一设定
        self.lamb = 0

    def forward(self, x):
        self.x = x
        self.y = np.array([np.dot(self.weights, xx) + self.bias for xx in x])
        return self.y

    def backward(self, d):
        ddw = [np.dot(dd, v.T) for dd, v in zip(d, self.x)]
        self.dw = np.sum(ddw, axis=0) / self.x.shape[0]
        self.dx = np.array([np.dot(self.weights.T, dd) for dd in d])
        self.weights -= self.lr * (self.dw + self.lamb * self.weights / self.x.shape[0])
        self.bias -= self.lr * np.sum(d, axis=0) / self.x.shape[0]
        return self.dx


class Accuracy:
    def __init__(self):
        self.type = 'accuracy'

    def forward(self, x, lable):
        self.accuracy = np.sum(np.argmax(x, axis=1), axis=1) == lable
        self.accuracy = np.sum(self.accuracy)
        print 'accuracy:', 1.0 * self.accuracy / len(x)

    def backward(self):
        raise Exception('Accuracy has no backward')


class Softmax:
    def __init__(self):
        self.type = 'softmax'

    def forward(self, x):
        self.x = x
        exp = np.exp(x)
        expsum = np.sum(exp, axis=1)
        self.y = np.array([a / b for a, b in zip(exp, expsum)])
        return self.y

    def backward(self):
        raise Exception('Softmax layer can not back-propagate')

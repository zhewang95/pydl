# Created by wz on 17-3-23.
# encoding=utf-8
import numpy as np


class FullyConnect:
    def __init__(self, l_in, l_out, lr, xavier=True):
        self.weights = np.random.randn(l_out, l_in)
        if xavier:
            self.weights /= np.sqrt(l_in)  # todo: read the Xavier paper
        self.bias = np.random.randn(l_out, 1)
        self.lr = lr

    def forward(self, vin):
        self.vin = vin
        self.vout = np.array([np.dot(self.weights, v) + self.bias for v in vin])
        return self.vout

    def backward(self, d):
        ddw = [np.dot(dd, v.T) for dd, v in zip(d, self.vin)]
        self.dw = np.sum(ddw, axis=0) / self.vin.shape[0]
        self.dvin = np.array([np.dot(self.weights.T, dd) for dd in d])
        self.weights -= self.lr * self.dw
        self.bias -= self.lr * np.sum(d, axis=0)
        return self.dvin


class Accuracy:
    def __init__(self):
        pass

    def forward(self, vin, lable):
        self.accuracy = np.sum(np.argmax(vin, axis=1), axis=1) == lable
        self.accuracy = np.sum(self.accuracy)
        print 'accuracy:', 1.0 * self.accuracy / len(vin)

    def backward(self):
        raise Exception('Accuracy has no backward')

# Created by wz on 17-3-23.
# encoding=utf-8
import numpy as np

class FullyConnect:
    def __init__(self, l_in, l_out, lr, xavier=True):
        self.weights = np.random.randn(l_out, l_in)
        if xavier:
            self.weights /= np.sqrt(l_in) # todo: read the Xavier paper
        self.bias = np.random.randn(l_out, 1)
        self.lr = lr

    def forward(self, vin):
        self.vin = vin
        self.vout = [np.dot(self.weights, v) + self.bias for v in vin]
        return self.vout

    def backward(self, d):
        self.ddw = [np.dot(d, v.T) for v in self.vin]
        self.dw = np.sum(self.ddw, axis=0) / self.vin.shape[0]
        self.dvin = np.dot(self.weights.T, d)
        self.weights -= self.lr * self.dw
        self.bias -= self.lr * d
        return self.dvin
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
        #print self.weights[0][0],self.dw[0][0]*self.lr,d[0][0]
        self.weights -= self.lr * self.dw
        self.bias -= self.lr * np.sum(d,axis=0)
        return self.dvin

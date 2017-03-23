# Created by wz on 17-3-23.
# encoding=utf-8
import numpy as np

class ReLU:
    def __init__(self, l_in):
        self.l_in = l_in

    def forward(self, vin):
        self.vin = vin
        self.vout = np.select([vin > 0], [vin], 0)
        return self.vout

    def backward(self, d):
        return np.select([self.vin > 0], [d], 0)
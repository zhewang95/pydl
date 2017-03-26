# Created by wz on 17-3-23.
# encoding=utf-8
import numpy as np

class ReLU:
    def __init__(self):
        pass

    def forward(self, vin):
        self.vin = vin
        self.vout = np.select([vin > 0], [vin], 0)
        return self.vout

    def backward(self, d):
        return np.select([self.vin > 0], [d], 0)


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, vin):
        self.vin = vin
        self.vout = 1.0 / (1 + np.exp(-vin))
        return self.vout

    def backward(self, d):
        e=np.exp(-self.vin)
        self.dvin= d*(-e)/np.square(1+e)
        return self.dvin

class TanH:
    def __init__(self):
        pass

    def forward(self,vin):
        self.vin=vin
        evinp=np.exp(vin)
        evinn=np.exp(-vin)
        self.vout=(evinp-evinn)/(evinp+evinn)
        return self.vout

    def backward(self,d):
        edp=np.exp(d)
        edn=np.exp(-d)
        tanh=(edp-edn)/(edp+edn)
        self.dvin=1-tanh*tanh
        return self.dvin

def test():
    relu = ReLU()
    print relu.forward(np.array([[1, -3], [100, -1000]]))
    print relu.backward(np.array([[1, -3], [100, -1000]]))


if __name__ == '__main__':
    test()

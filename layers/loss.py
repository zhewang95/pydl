# Created by wz on 17-3-24.
# encoding=utf-8
import numpy as np


class SoftMaxLoss:
    def __init__(self):
        self.type = 'loss'

    def forward(self, vin, lable):
        expsum = np.sum(np.exp(vin), axis=1)
        self.lable = lable
        self.prob = np.array([np.exp(v) / s for v, s in zip(vin, expsum)])
        self.loss = np.nan_to_num(-np.log(np.array([p[i] for p, i in zip(self.prob, lable)])))
        return np.sum(self.loss) / lable.shape[0]

    def backward(self):
        self.dvin = np.copy(self.prob)
        for v, l in zip(self.dvin, self.lable):
            v[l] -= 1
        return self.dvin


class CrossEntropyLoss:
    def __init__(self):
        self.type = 'loss'

    def forward(self, vin, lable):
        self.vin = vin
        self.lable = np.zeros_like(vin)
        for a, b in zip(self.lable, lable):
            a[b] = 1.0
        self.loss = np.nan_to_num(-(self.lable * np.log(vin) + (1 - self.lable) * np.log(1 - vin)))
        self.loss = np.sum(self.loss, axis=1)
        return np.sum(self.loss, axis=0) / len(vin)

    def backward(self):
        self.d = (self.vin - self.lable) / self.vin / (1 - self.vin)
        return self.d


def test():
    sml = SoftMaxLoss()
    print sml.forward(np.array([[[2], [1]]]), np.array([0]))
    print sml.backward()


if __name__ == '__main__':
    test()

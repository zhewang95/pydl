# Created by wz on 17-3-24.
# encoding=utf-8
import numpy as np


class QuadraticLoss:
    def __init__(self):
        self.type = 'loss'
        self.lamb = 0.0

    def forward(self, x, label, hidden_layers):
        self.x = x
        self.label = np.zeros_like(x)
        for a, b in zip(self.label, label):
            a[b] = 1.0
        self.loss = np.sum(np.square(x - self.label)) / x.shape[0] / 2
        for layer in hidden_layers:
            if layer.type in ['fullyconnect']:
                self.loss += self.lamb * np.sum(np.square(layer.weights)) / 2 / x.shape[0]
        return self.loss

    def backward(self):
        self.dx = (self.x - self.label) / self.x.shape[0]
        return self.dx


class SoftmaxLoss:
    def __init__(self):
        self.type = 'loss'
        self.lamb = 0

    def forward(self, x, label, hidden_layers):
        exp = np.exp(x)
        expsum = np.sum(exp, axis=1)
        self.label = label
        self.prob = np.array([v / s for v, s in zip(exp, expsum)])
        self.loss = np.nan_to_num(-np.log(np.array([p[i] for p, i in zip(self.prob, label)])))
        self.loss = np.sum(self.loss) / label.shape[0]
        for layer in hidden_layers:
            if layer.type in ['fullyconnect']:
                self.loss += self.lamb * np.sum(np.square(layer.weights)) / 2 / x.shape[0]
        return self.loss

    def backward(self):
        self.dx = np.copy(self.prob)
        for v, l in zip(self.dx, self.label):
            v[l] -= 1
        return self.dx


class CrossEntropyLoss:
    def __init__(self):
        self.type = 'loss'
        self.lamb = 0

    def forward(self, x, label, hidden_layers):
        self.x = x
        self.label = np.zeros_like(x)  # 将数字代表的标签转换成向量标签
        for a, b in zip(self.label, label):
            a[b] = 1.0
        self.loss = np.nan_to_num(
            -(self.label * np.log(x + 10 ** -10) + (1 - self.label) * np.log(
                1 - x + 10 ** -10)))  # np.nan_to_num
        self.loss = np.sum(self.loss) / x.shape[0]
        for layer in hidden_layers:
            if layer.type in ['fullyconnect']:
                self.loss += self.lamb * np.sum(np.square(layer.weights)) / 2 / x.shape[0]
        return self.loss

    def backward(self):
        self.dx = (self.x - self.label) / self.x / (1 - self.x)
        return self.dx


class SigmoidCrossEntropyLoss:
    def __init__(self):
        self.type = 'loss'
        self.lamb = 0.0

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, x, label, hidden_layers):
        self.x = self.sigmoid(x)
        self.label = np.zeros_like(x)
        for a, b in zip(self.label, label):
            a[b] = 1.0
        self.loss = np.nan_to_num(
            -(self.label * np.log(self.x + 10 ** -10) + (1 - self.label) * np.log(
                1 - self.x + 10 ** -10)))
        self.loss = np.sum(self.loss) / x.shape[0]
        for layer in hidden_layers:
            if layer.type in ['fullyconnect']:
                self.loss += self.lamb * np.sum(np.square(layer.weights)) / 2 / x.shape[0]
        return self.loss

    def backward(self):
        self.dx = self.x - self.label
        return self.dx


def test():
    sml = SoftmaxLoss()
    print sml.forward(np.array([[[2], [1]]]), np.array([0]))
    print sml.backward()


if __name__ == '__main__':
    test()

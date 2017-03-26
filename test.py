# Created by wz on 17-3-26.
# encoding=utf-8
from layers.data import PickleData
from layers.misc import FullyConnect, Accuracy
from layers.activation import Sigmoid, ReLU
from layers.loss import SoftMaxLoss, CrossEntropyLoss
from solvers.solvers import NaiveSGD

if __name__ == '__main__':
    ls = []
    ls.append(PickleData('data/test1.pkl', 256))
    ls.append(FullyConnect(289, 100, 0.01))
    ls.append(ReLU())
    ls.append(FullyConnect(100, 26, 0.01))
    ls.append(ReLU())
    ls.append(SoftMaxLoss())
    accu = Accuracy()

    trainer = NaiveSGD(ls, 1000, accu)
    trainer.train()

# Created by wz on 17-3-26.
# encoding=utf-8
from layers.data import PickleData
from layers.misc import FullyConnect, Accuracy
from layers.activation import Sigmoid, ReLU
from layers.loss import SoftMaxLoss, CrossEntropyLoss
from solvers.solvers import NaiveSGD

if __name__ == '__main__':
    network = []
    network.append(PickleData('data/train.pkl', 512))
    network.append(PickleData('data/validate.pkl', 50000))
    network.append(FullyConnect(289, 100, 0.03))
    network.append(Sigmoid())
    network.append(FullyConnect(100, 26, 0.03))
    network.append(Sigmoid())
    network.append(SoftMaxLoss())
    accu = Accuracy()

    trainer = NaiveSGD(network, 1000, accu)
    trainer.train()

# Created by wz on 17-3-26.
# encoding=utf-8
from layers.data import *
from layers.misc import *
from layers.activation import *
from layers.loss import *
from solvers.solvers import *

if __name__ == '__main__':
    network = []
    network.append(PickleData('data/train.pkl', 1024))
    network.append(PickleData('data/test.pkl', 10000))
    network.append(FullyConnect(17 * 17, 20))
    network.append(ReLU())
    network.append(FullyConnect(20, 26))
    network.append(SoftmaxLoss())
    accuracy = Accuracy()

    trainer = NaiveSGD(network, 100, accuracy=accuracy, lr=1, lamb=0.001)
    trainer.train()

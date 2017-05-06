# Created by wz on 17-3-26.
# encoding=utf-8
from layers.data import PickleData
from layers.misc import FullyConnect, Accuracy
from layers.activation import Sigmoid, ReLU, TanH
from layers.loss import SoftMaxLoss, CrossEntropyLoss, QuadraticLoss, SigmoidCrossEntropyLoss
from solvers.solvers import NaiveSGD

if __name__ == '__main__':
    network = []
    network.append(PickleData('data/train.pkl', 1024))
    network.append(PickleData('data/test.pkl', 10000))
    network.append(FullyConnect(17 * 17, 20))
    network.append(Sigmoid())
    network.append(FullyConnect(20, 26))
    network.append(Sigmoid())
    network.append(QuadraticLoss())
    accuracy = Accuracy()

    trainer = NaiveSGD(network, 100, accuracy=accuracy, lr=1000, lamb=0.0001)
    trainer.train()

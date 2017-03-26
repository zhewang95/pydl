# Created by wz on 17-3-23.
# encoding=utf-8
import cPickle, random
import numpy as np


class PickleData:
    def __init__(self, source, batch_size):
        with open(source) as f:
            data = cPickle.load(f)
        self.data = data['data']
        self.labels = data['labels']
        self.batch_size = batch_size
        self.pos = 0
        self.l = len(self.data)
        assert (self.l > 0)
        assert (self.l == len(self.labels))
        #assert (len(self.data.shape) == 3)

    def forward(self):
        front = self.pos
        if self.pos + self.batch_size >= self.l:
            rear = self.l
            self.pos = 0
            ran=range(self.l)
            np.random.shuffle(ran)
            self.data=self.data[ran]
            self.labels=self.labels[ran]
        else:
            rear = self.pos + self.batch_size
            self.pos += self.batch_size
        return self.data[front:rear], self.labels[front:rear], self.pos

    def backword(self):
        pass

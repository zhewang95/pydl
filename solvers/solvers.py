# Created by wz on 17-3-23.
# encoding=utf-8

class NaiveSGD:
    def __init__(self, layers, epochs, accuracy):
        self.layers = layers
        self.epochs = epochs
        self.accuracy = accuracy

    def train(self):
        dlayer = self.layers[0]
        llayer = self.layers[-1]
        for i in xrange(self.epochs):
            print 'epoch:', i
            loss = 0.0
            accuracy = 0.0
            count = 0
            while True:
                data, lable, pos = dlayer.forward()
                for l in self.layers[1:-1:1]:
                    data = l.forward(data)
                if (pos == 0):
                    self.accuracy.forward(data, lable)
                l = llayer.forward(data, lable)
                loss += l
                count += 1
                d = llayer.backward()
                for l in self.layers[-2:0:-1]:
                    d = l.backward(d)
                if pos == 0:
                    break
            print loss / count

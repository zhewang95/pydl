# Created by wz on 17-3-23.
# encoding=utf-8

class NaiveSGD:
    def __init__(self, layers, epochs, accuracy):
        self.dlayer = layers[0]
        self.llayer = layers[-1]
        if layers[1].type != 'data':
            self.hidden_layers = layers[1:-1]
            self.tdlayer = None
        else:
            self.hidden_layers = layers[2:-1]
            self.tdlayer = layers[1]

        self.epochs = epochs
        self.accuracy = accuracy

    def forward(self, data):
        for l in self.hidden_layers:
            data = l.forward(data)
        return data

    def backward(self, d):
        for l in self.hidden_layers[::-1]:
            d = l.backward(d)

    def train(self):
        for i in xrange(self.epochs):
            print 'epoch:', i
            loss = 0.0
            count = 0
            while True:
                data, label, pos = self.dlayer.forward()
                data = self.forward(data)
                l = self.llayer.forward(data, label)

                loss += l
                count += 1

                d = self.llayer.backward()
                self.backward(d)
                if pos == 0:  # 每轮训练结束后进行测试
                    if self.tdlayer:
                        data, label, _ = self.tdlayer.forward()
                        data = self.forward(data)
                        self.accuracy.forward(data, label)
                    break

            print 'loss:', loss / count

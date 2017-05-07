# Created by wz on 17-3-23.
# encoding=utf-8
import numpy as np


# import scipy.ndimage as ndimage
# import cc.cc

class Convolution:
    # c: channel, w: width, h: height, s_x: stride_x, s_y: stride_y,
    # p_x: padding_x, p_y: padding_y, k_x:kernel_x, k_y: kernel_y, f: feature
    def __init__(self, c, w, h, s_x, s_y, p_x, p_y, k_x, k_y, f):
        self.type = 'convolution'
        self.c, self.w, self.h = c, w, h
        self.s_x, self.s_y = s_x, s_y
        self.p_x, self.p_y = p_x, p_y
        self.k_x, self.k_y = k_x, k_y
        self.f = f
        assert ((w + 2 * p_x - k_x) % s_x == 0)
        assert ((h + 2 * p_y - k_y) % s_y == 0)
        self.ow = (w + 2 * p_x - k_x) / s_x + 1
        self.oh = (h + 2 * p_y - k_y) / s_y + 1
        self.weights = np.random.randn(f, c * k_x * k_y) / np.sqrt(c * k_x * k_y)
        self.bias = np.random.randn(f)
        self.lr, self.lamb = 0.0, 0.0

    def pad(self, x):
        s = x.shape
        zero = np.zeros((s[0], s[1], s[2] + 2 * self.p_x, s[3] + 2 * self.p_y))
        zero[:, :, self.p_x:self.p_x + self.w, self.p_y:self.p_y + self.h] = x
        return zero

    # more memory for less time
    def eval(self, x):
        ww = self.w + 2 * self.p_x - self.k_x + 1
        hh = self.h + 2 * self.p_y - self.k_y + 1
        ret = np.array([[[np.ravel(xx[:, a:a + self.k_x, b:b + self.k_y]) for b in range(0, hh, self.s_y)]
                         for a in range(0, ww, self.s_x)] for xx in x])
        return ret

    def de_eval_pad(self, x):
        a1, a2, a3, a4 = x.shape
        x = x.reshape(a1, a2, a3, self.c, self.k_x, self.k_y)
        ret = np.zeros_like(self.px)
        for i in range(a1):
            for j in range(a2):
                w = j * self.s_x
                for t in range(a3):
                    h = t * self.s_y
                    ret[i][:, w:w + self.k_x, h:h + self.k_y] += x[i][j][t]
        return ret[:, :, self.p_x:self.p_x + self.w, self.p_y:self.p_y + self.h]

    def forward(self, x):
        # for-loop implement, too slow
        # ix = [range(i, i + self.k_x) for i in range(0, self.w + 2 * self.p_x - self.k_x + 1, self.s_x)]
        # iy = [range(i, i + self.k_y) for i in range(0, self.h + 2 * self.p_y - self.k_y + 1, self.s_y)]
        # self.y = np.array([[[[np.sum(xx[:, ix[i], iy[j]] * w) + b for j in range(len(iy))] for i in range(len(ix))]
        #                    for w, b in zip(self.weights, self.bias)] for xx in px])
        # self.y = np.array([[[ndimage.convolve(xx, w) + b] for w, b in zip(self.weights, self.bias)] for xx in px])
        # print self.y.shape
        assert (x.shape[1] == self.c)
        assert (x.shape[2] == self.w)
        assert (x.shape[3] == self.h)
        self.px = self.pad(x)
        self.evalx = self.eval(self.px)
        self.y = (self.evalx.dot(self.weights.T) + self.bias).transpose(0, 3, 1, 2)
        return self.y

    def backward(self, d):
        a1, a2, a3, a4 = d.shape
        self.ddx = np.array([dd.T.dot(self.weights) for dd in d])
        self.dx = self.de_eval_pad(self.ddx)
        d = d.reshape(a1, a2, a3 * a4)
        self.evalx = self.evalx.reshape(a1, a3 * a4, self.c * self.k_x * self.k_y)
        self.dw = np.sum([dd.dot(x) for x, dd in zip(self.evalx, d)], axis=0) / a1 / a3 / a4
        self.db = np.sum(d, axis=(0, 2)) / a1 / a3 / a4

        self.weights -= self.lr * (self.dw + self.lamb * np.sum(np.square(self.weights)) / a1)
        self.bias -= self.lr * self.db
        return self.dx


if __name__ == "__main__":
    c = Convolution(1, 17, 17, 1, 1, 1, 1, 3, 3, 32)
    y1 = c.forward(np.ones((100, 1, 17, 17)))
    print y1.shape
    y2 = c.backward(y1)
    print y2.shape
    # p = c.pad(np.ones((100, 1, 2, 2)))
    # print p

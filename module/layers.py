import collections
from copy import deepcopy

import cupy as np
import cupyx

from module.functions import *


class MatMul :
    def __init__(self, W) :
        self.params, self.grads = [W], [np.zeros_like(W)]
        self.cache = None

    def forward(self, x) :
        W, = self.params
        self.cache = x
        out = np.dot(x, W)
        return out

    def backward(self, dout) :
        W, = self.params
        x = self.cache
        dx = np.dot(dout, W.T)
        dW = np.dot(x.T, dout)
        self.grads[0][...] = dW
        return dx


class ReLU :
    def __init__(self) :
        self.params, self.grads = [], []
        self.mask = None

    def forward(self, x) :
        self.mask = x <= 0
        x[self.mask] = 0.
        return x

    def backward(self, dx) :
        dx[self.mask] = 0.
        self.mask = None
        return dx


class Embedding :
    def __init__(self, W) :
        self.params, self.grads = [W], [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx) :
        W, = self.params
        # |idx| = (bs, )
        self.idx = idx
        return W[idx]

    def backward(self, dout) :
        W, = self.params
        dW = np.zeros_like(W, dtype='f')
        # if self.idx.ndim == 3:
        #   dout = np.expand_dims(dout, axis=1)
        cupyx.scatter_add(dW, self.idx, dout)
        self.grads[0][...] = dW  # dW가 계속 더해지면서 기울기가 발산
        return None


class Sigmoid :
    def __init__(self) :
        self.params, self.grads = [], []
        self.loss = None
        self.y_hat = None
        self.y = None

    def forward(self, score, y) :
        self.y = y
        self.y_hat = sigmoid(score)
        self.loss = binary_cross_entropy(self.y_hat, self.y)

        return self.loss

    def backward(self, dout=1) :
        batch_size = self.y.shape[0]
        dx = (self.y_hat - self.y) * dout / batch_size
        return dx


class Softmax :
    def __init__(self) :
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, score, y) :
        score -= np.max(score, axis=1, keepdims=True)
        y_hat = np.exp(score) / np.sum(np.exp(score), axis=1, keepdims=True)
        loss = -np.log(y_hat[np.arange(len(y)), y]).sum() / len(y)
        self.cache = y, y_hat
        return loss

    def backward(self, dout=1.) :
        y, y_hat = self.cache
        dout /= len(y)
        y_hat[np.arange(len(y)), y] -= 1
        dscore = y_hat * dout
        return dscore


# class Softmax:
#   def __init__(self):
#     self.params, self.grads = [], []
#     self.cache = None

#   def forward(self, score, y):
#     score -= np.max(score, axis=1, keepdims=True)
#     y_hat = np.exp(score) / np.sum(np.exp(score), axis=1, keepdims=True)
#     loss = -(y * np.log(y_hat)).sum() / len(y)
#     self.cache = y, y_hat
#     return loss

#   def backward(self, dout=1.):
#     y, y_hat = self.cache
#     dout /= len(y)
#     dscore = (y_hat - y) * dout
#     return dscore


class NegativeSampling :
    def __init__(self, corpus, power, vocab_size, sample_size, batch_size, exclude_target=False) :
        self.vocab_size = vocab_size
        self.sample_size = sample_size
        self.ps = np.zeros(vocab_size, dtype='f')
        for k, v in collections.Counter(corpus).items() :
            self.ps[k] += float(v)

        self.ps **= power
        self.ps /= self.ps.sum(keepdims=True)

        self.exclude_target = exclude_target

    def sampling(self, target) :
        batch_size = target.shape[0]
        neg_samps = []

        if self.exclude_target :
            ps = deepcopy(self.ps)
            # |ps| = (batch size, vocab size)
            ps[target] = 0.
            ps /= ps.sum()
            neg_samps = np.random.choice(self.vocab_size,
                                         size=(batch_size, self.sample_size),
                                         replace=True,
                                         p=ps).astype('i')

        else :
            neg_samps = np.random.choice(self.vocab_size,
                                         size=(batch_size, self.sample_size),
                                         replace=True,
                                         p=self.ps)
        samples = np.concatenate([target.reshape(-1, 1), neg_samps], axis=1)
        labels = np.concatenate([np.ones((batch_size, 1)),
                                 np.zeros((batch_size, self.sample_size))], axis=1)
        return samples, labels
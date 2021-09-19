import cupy as np

from module.layers import *


class SimpleCBOW :
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 window_size) :
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        W_in = np.random.randn(vocab_size, hidden_size).astype('f') * (2. / np.sqrt(vocab_size + hidden_size))
        W_out = np.random.randn(hidden_size, vocab_size).astype('f') * (2. / np.sqrt(hidden_size + vocab_size))

        self.layers = [MatMul(W_in) for _ in range(2 * window_size)] + [MatMul(W_out), Softmax()]
        self.params, self.grads = [], []
        for layer in self.layers :
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, x, y) :
        h = 0.
        for i, layer in enumerate(self.layers[:2 * self.window_size]) :
            h += layer.forward(x[:, i])

        h /= 2 * self.window_size

        score = self.layers[-2].forward(h)
        loss = self.layers[-1].forward(score, y)
        return loss

    def backward(self, dout=1.) :
        for layer in reversed(self.layers[-2 :]) :
            dout = layer.backward(dout)

        dout /= 2 * self.window_size

        for layer in self.layers[:2 * self.window_size] :
            layer.backward(dout)

        return None


class CBOW :
    def __init__(self,
                 corpus,
                 vocab_size,
                 hidden_size,
                 window_size,
                 batch_size,
                 sample_size,
                 power,
                 exclude_target) :
        self.window_size = window_size
        self.sample_size = sample_size

        W_in = np.random.randn(vocab_size, hidden_size).astype('f') * (2. / np.sqrt(vocab_size + hidden_size))
        W_out = np.random.randn(vocab_size, hidden_size).astype('f') * (2. / np.sqrt(vocab_size + hidden_size))

        self.W_in = W_in
        self.W_out = W_out

        self.sampler = NegativeSampling(corpus, power, vocab_size, sample_size, batch_size, exclude_target)
        self.layers = [Embedding(W_in), Embedding(W_out), Sigmoid()]

        self.params, self.grads = [], []
        for layer in self.layers :
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, contexts, target) :
        h = self.layers[0].forward(contexts).sum(axis=1) / (self.window_size * 2)
        samps, labs = self.sampler.sampling(target)
        embs = self.layers[1].forward(samps)
        dot = (embs * np.expand_dims(h, axis=1)).sum(axis=2)
        loss = self.layers[2].forward(dot, labs)
        self.cache = h, embs
        return loss

    def backward(self, dout=1) :
        h, embs = self.cache

        ddot = self.layers[2].backward(dout)
        dembs = np.expand_dims(ddot, axis=2) * np.expand_dims(h, axis=1)
        self.layers[1].backward(dembs)
        dh = (np.expand_dims(ddot, axis=2) * embs).sum(axis=1)
        dh = np.expand_dims(dh, axis=1).repeat(2 * self.window_size, axis=1)
        dh /= (self.window_size * 2)
        self.layers[0].backward(dh)
        return None


class Skipgram :
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 window_size) :
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        W_in = np.random.randn(vocab_size, hidden_size).astype('f') * (2. / np.sqrt(vocab_size + hidden_size))
        W_out = np.random.randn(hidden_size, vocab_size).astype('f') * (2. / np.sqrt(vocab_size + hidden_size))

        self.W_in = W_in
        self.W_out = W_out

        self.layers = [Embedding(W_in), MatMul(W_out)]
        self.loss = [Softmax() for _ in range(2 * self.window_size)]

        self.params, self.grads = [], []
        for layer in self.layers :
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, x, y) :
        loss = 0.
        h = self.layers[0].forward(x)
        score = self.layers[1].forward(h)
        for i, layer in enumerate(self.loss) :
            loss += layer.forward(score, y[:, i])
        return loss

    def backward(self, dout=1.) :
        dh = 0.
        for layer in self.loss :
            dscore = layer.backward(dout)
            dh += self.layers[1].backward(dscore)
        self.layers[0].backward(dh)
        return None

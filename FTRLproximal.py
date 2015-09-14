from __future__ import division

import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


class FTRLProximal(object):
    def __init__(self, alpha=0.1, beta=1., l1=0., l2=0.):
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        self._z = {}
        self._n = {}
        self._round = 0
        self._history = []
        self._bias = 0
        self._nbias = 0

    @property
    def weights(self):
        w = {}
        a = self.alpha
        b = self.beta
        n = self._n
        l1 = self.l1
        l2 = self.l2
        for k, zi in self._z.iteritems():
            if np.abs(zi) <= l1:
                w[k] = 0
            else:
                w[k] = -(zi - l1 * np.sign(zi)) / (l2 + (b + np.sqrt(n[k])) / a)
        return w

    def _linear_predictor(self, x, w):
        return self._bias + sum(w.get(k, 0) * v for k, v in x.iteritems())

    def _predict(self, x, w):
        return sigmoid(self._linear_predictor(x, w))

    def predict(self, x):
        w = self.weights
        if type(x) is list:
            return [self._predict(xi, w) for xi in x]
        else:
            return self._predict(x, w)

    def fit(self, x, y):
        w = self.weights
        pred = self._predict(x, w)
        self._update(x, y, pred, w)
        self._history.append((pred, y))

    def _update(self, x, y, p, w):
        err = p - y
        for k, xi in x.iteritems():
            gi = err*xi
            gg = gi**2
            ni = self._n.get(k, 0)
            si = (np.sqrt(ni + gg) - np.sqrt(ni)) / self.alpha
            self._z[k] = self._z.get(k, 0) + gi - si * w.get(k, 0)
            self._n[k] = self._n.get(k, 0) + gg
        self._bias -= self.alpha * err / (self.beta + np.sqrt(self._nbias))
        self._nbias += err**2
        self._round += 1

    def uncertainty_score(self, x):  # This is a work in progress
        return self.alpha * sum(v / np.sqrt(self._n.get(k, 0))
                                for k, v in x.iteritems())


class FTRLProximalSVM(FTRLProximal):
    def _predict(self, x, w):
        return self._linear_predictor(x, w)

    def _update(self, x, y, p, w):
        err = -np.max(1 - p*y, 0)*y
        for k, xi in x.iteritems():
            gi = err*xi
            gg = gi**2
            ni = self._n.get(k, 0)
            si = (np.sqrt(ni + gg) - np.sqrt(ni)) / self.alpha
            self._z[k] = self._z.get(k, 0) + gi - si * w.get(k, 0)
            self._n[k] = self._n.get(k, 0) + gg
        self._bias -= self.alpha * err / (self.beta + np.sqrt(self._nbias))
        self._nbias += err**2
        self._round += 1
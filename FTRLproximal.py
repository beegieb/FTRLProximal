from __future__ import division
from collections import deque

import numpy as np


def sigmoid(x):
    """
    Computes the sigmoid function on x

    :param x: a numeric type or numpy-friendly array
    :returns: the sigmoid function applied to x, if x is numpy-friendly,
              then a numpy array with sigmoid applied to each element
    """
    return 1/(1 + np.exp(-x))


class FTRLProximal(object):
    """
    A pure python implementation of the FTRLProximal algorithm described in
    ["Ad Click Prediction: a View from the Trenches"][1] and
    ["Follow-the-Regularized-Leader and Mirror Descent: Equivalence Theorems and L1 Regularization"][2]

    This implementation relies on native python data-types with the intention of making the learned parameters
    human readable

    [1][http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf]
    [2][http://jmlr.org/proceedings/papers/v15/mcmahan11b/mcmahan11b.pdf]
    """
    def __init__(self, alpha=1., beta=1., l1=0., l2=0., history_len=None):
        """
        :param alpha: Parameter used to control scaling of per-weight learning rates. (Default 1.0)
                      Can be loosely thought of as a learning rate in Gradient Descent.
                      This should be tuned with validation.
        :param beta: Parameter used to stabalize the magnitude of early per-weight learning rates (Default 1.0)
                     Learning rates are approximately defined as alpha / (beta + stuff). At the beginning of training
                     if 'stuff' is small, the learning rate for parameters will be large, beta is used to stabalize the
                     denominator. Leaving this at 1.0 should be good enough.
        :param l1: The magnitude of L1 regularization (Default 0.0)
        :param l2: The magnitude of L2 regularization (Default 0.0)
        """
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        self._z = {}
        self._n = {}
        self._round = 0
        self._history = deque(maxlen=history_len)
        self._bias = 0
        self._nbias = 0

    @property
    def bias(self):
        """
        :returns: The learned bias of the model
        """
        return self._bias

    @property
    def weights(self):
        """
        :returns: The learned coefficients of the model
        """
        return self._weights(self._z.keys())

    def _weights(self, keys):
        w = {}
        a = self.alpha
        b = self.beta
        n = self._n
        l1 = self.l1
        l2 = self.l2
        for k in keys:
            zi = self._z.get(k, 0)
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
        """
        Make a prediction on data x using the learned parameters of the model

        :param x: a dict or list of dicts. The dict is an individaul datapoint represented in a sparse format
                  if a list of dicts, each element should be datapoints represented in sparse formats
        :returns: if x is a singular dict, returns the models prediction on x,
                 if x is a list, returns a list of element-wise predictions
        """
        w = self.weights  # we only want to generate the weights once
        if type(x) is list:  # TODO: Allow any iterable to be passed for predictions
            return [self._predict(xi, w) for xi in x]
        else:
            return self._predict(x, w)

    def fit(self, x, y):
        """
        Fit the model on an x, y datapoint pair
        :param x: a sparse dict with format {'feature1_name': value, ...}
                  featureN_name is a human readable name, value is numeric
                  to save on computation, any zero-values in x should be omitted from the dict
        :param y: the binary classification label for x
        :returns: self
        """
        w = self._weights(x.keys())
        pred = self._predict(x, w)
        self._update(x, y, pred, w)
        self._history.append((pred, y))
        return self

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
        """
        Compute the uncertainty score described in Google's paper

        The goal of the uncertainty score is to be a heuristic way of quantifying uncertainty

        NOTE: This is still a work-in-progress and currently relies on the assumption that each value of x
              has magnitude <= 1.0

        :param x: a dict representation of the datapoint
        :returns: an uncertainty score of the prediction
        """
        return self.alpha * sum(np.abs(v) / np.sqrt(self._n.get(k, 0))
                                for k, v in x.iteritems())


class FTRLProximalSVM(FTRLProximal):
    """
    An implementation of the FTRLProximal algorithm using the L2-SVM as the objective function for training

    The only differences from the logistic implementation is the objective function used, and the ouput range of
    predictions

    Predictions, p, from this model are on the interval (-inf, inf), and sign(p) should be used as the classification
    label
    """
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
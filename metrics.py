from __future__ import print_function

import sys
import numpy as np


def masked_mean(val,
                mask,
                axis = None):
    return np.sum(val * mask, axis=axis) / np.sum(mask, axis=axis)


def cosine_similarity(x):
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    sim = np.zeros([x.shape[0], x.shape[0]], dtype=x.dtype)
    x.dot(x.T, out=sim)
    return sim


def nearest_neighbors(emb, k):
    sim = cosine_similarity(emb)
    return (-sim).argsort(axis=-1)[:, 1:(k + 1)]


def mean_nearest_neighbor_overlap(x, y, k):
    x_nn = nearest_neighbors(x, k)
    y_nn = nearest_neighbors(y, k)
    assert x_nn.shape[0] == y_nn.shape[0]

    res = 0
    for i in range(x_nn.shape[0]):
        res += np.intersect1d(x_nn[i], y_nn[i]).shape[0] / float(x_nn.shape[1])
    return res / x_nn.shape[0]


def pearson(x, y, mask):
    x_sim = cosine_similarity(x)
    y_sim = cosine_similarity(y)

    x_sim_mean = masked_mean(x_sim, mask)
    y_sim_mean = masked_mean(y_sim, mask)

    x_sim -= x_sim_mean
    y_sim -= y_sim_mean

    x_sim_sigma = np.sqrt(masked_mean(x_sim * x_sim, mask))
    y_sim_sigma = np.sqrt(masked_mean(y_sim * y_sim, mask))

    res = masked_mean(x_sim * y_sim, mask) / (x_sim_sigma * y_sim_sigma)
    return res

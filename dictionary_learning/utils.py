"""
Created on Thu Jan 11 15:42:40 2018

@author: Matin Kheirkhahan (matinkheirkhahan@ufl.edu)
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

class Utils(object):
    """
    A class containing functions that are used in dictionary learning algorithms.
    This class specifically provides functions which are parameters (options) used in the algorithms used in dictionary learning.
    """

    @staticmethod
    def __traceback(D):
        """
        Provides the mapping between rows and columns of the given matrix.
        :param D: matrix formed of two vectors to be compared.
        :return: the mapping between the two vectors.
        """
        i, j = np.array(D.shape) - 2
        p, q = [i], [j]
        while ((i > 0) or (j > 0)):
            tb = np.argmin((D[i, j], D[i, j+1], D[i+1, j]))
            if (tb == 0):
                i -= 1
                j -= 1
            elif (tb == 1):
                i -= 1
            else: # (tb == 2):
                j -= 1
            p.insert(0, i)
            q.insert(0, j)
        return np.array(p), np.array(q)

    @staticmethod
    def __fastdtw(x, y, dist='euclidean', show_path=False):
        """
        Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
        Instead of iterating through each element and calculating each distance,
        this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
        :param array x: N1*M array
        :param array y: N2*M array
        :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
        If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
        :param show_path: if True, it shows the assignment between points of the two passed vectors. (Default: False| to expedite the process)
        :return: Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
        """
        assert len(x)
        assert len(y)
        if np.ndim(x) == 1:
            x = x.reshape(-1, 1)
        if np.ndim(y) == 1:
            y = y.reshape(-1, 1)
        r, c = len(x), len(y)
        D0 = np.zeros((r + 1, c + 1))
        D0[0, 1:] = np.inf
        D0[1:, 0] = np.inf
        D1 = D0[1:, 1:]
        D0[1:, 1:] = cdist(x, y, dist)
        C = D1.copy()
        for i in range(r):
            for j in range(c):
                D1[i, j] += min(D0[i, j], D0[i, j + 1], D0[i + 1, j])

        path = None
        if show_path:
            if len(x) == 1:
                path = np.zeros(len(y)), range(len(y))
            elif len(y) == 1:
                path = range(len(x)), np.zeros(len(x))
            else:
                path = Utils.__traceback(D0)
        return D1[-1, -1] / sum(D1.shape), C, D1, path

    @staticmethod
    def dtw(x, y, dist='euclidean'):
        d, _, _, _ = Utils.__fastdtw(x, y, dist=dist)
        return d
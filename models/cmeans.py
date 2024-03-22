"""
cmeans.py : Fuzzy C-means clustering algorithm.
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_random_state


def _cmeans0(data, u_old, c, m, metric):
    """
    Single step in generic fuzzy c-means clustering algorithm.

    Modified from Ross, Fuzzy Logic w/Engineering Applications (2010),
    pages 352-353, equations 10.28 - 10.35.

    Parameters inherited from cmeans()
    """
    # Normalizing, then eliminating any potential zero values.
    u_old = normalize_columns(u_old)
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old ** m

    # Calculate cluster centers
    data = data.T
    cntr = um.dot(data) / np.atleast_2d(um.sum(axis=1)).T

    d = _distance(data, cntr, metric)
    d = np.fmax(d, np.finfo(np.float64).eps)

    jm = (um * d ** 2).sum()

    u = normalize_power_columns(d, - 2. / (m - 1))

    return cntr, u, jm, d


def _distance(data, centers, metric='euclidean'):
    """
    Euclidean distance from each point to each cluster center.

    Parameters
    ----------
    data : 2d array (N x Q)
        Data to be analyzed. There are N data points.
    centers : 2d array (C x Q)
        Cluster centers. There are C clusters, with Q features.
    metric: string
        By default is set to euclidean. Passes any option accepted by
        ``scipy.spatial.distance.cdist``.
    Returns
    -------
    dist : 2d array (C x N)
        Euclidean distance from each point, to each cluster center.

    See Also
    --------
    scipy.spatial.distance.cdist
    """
    return cdist(data, centers, metric=metric).T

def _fp_coeff(u):
    """
    Fuzzy partition coefficient `fpc` relative to fuzzy c-partitioned
    matrix `u`. Measures 'fuzziness' in partitioned clustering.

    Parameters
    ----------
    u : 2d array (C, N)
        Fuzzy c-partitioned matrix; N = number of data points and C = number
        of clusters.

    Returns
    -------
    fpc : float
        Fuzzy partition coefficient.

    """
    n = u.shape[1]

    return np.trace(u.dot(u.T)) / float(n)

def _cmeans_transform0(test_data, cntr, u_old, c, m, metric):
    """
    Single step in fuzzy c-means prediction algorithm. Clustering algorithm
    modified from Ross, Fuzzy Logic w/Engineering Applications (2010)
    p.352-353, equations 10.28 - 10.35, but this method to generate fuzzy
    predictions was independently derived by Josh Warner.

    Parameters inherited from cmeans()

    Very similar to initial clustering, except `cntr` is not updated, thus
    the new test data are forced into known (trained) clusters.
    """
    # Normalizing, then eliminating any potential zero values.
    u_old = normalize_columns(u_old)
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old ** m
    test_data = test_data.T

    # For prediction, we do not recalculate cluster centers. The test_data is
    # forced to conform to the prior clustering.

    d = _distance(test_data, cntr, metric)
    d = np.fmax(d, np.finfo(np.float64).eps)

    jm = (um * d ** 2).sum()

    u = normalize_power_columns(d, - 2. / (m - 1))

    return u, jm, d


def normalize_columns(columns):
    """
    Normalize columns of matrix.

    Parameters
    ----------
    columns : 2d array (M x N)
        Matrix with columns

    Returns
    -------
    normalized_columns : 2d array (M x N)
        columns/np.sum(columns, axis=0, keepdims=1)
    """

    # broadcast sum over columns
    normalized_columns = columns / np.sum(columns, axis=0, keepdims=True)

    return normalized_columns


def normalize_power_columns(x, exponent):
    """
    Calculate normalize_columns(x**exponent)
    in a numerically safe manner.

    Parameters
    ----------
    x : 2d array (M x N)
        Matrix with columns
    n : float
        Exponent

    Returns
    -------
    result : 2d array (M x N)
        normalize_columns(x**n) but safe

    """

    assert np.all(x >= 0.0)

    x = x.astype(np.float64)

    # values in range [0, 1]
    x = x / np.max(x, axis=0, keepdims=True)

    # values in range [eps, 1]
    x = np.fmax(x, np.finfo(x.dtype).eps)

    if exponent < 0:
        # values in range [1, 1/eps]
        x /= np.min(x, axis=0, keepdims=True)

        # values in range [1, (1/eps)**exponent] where exponent < 0
        # this line might trigger an underflow warning
        # if (1/eps)**exponent becomes zero, but that's ok
        x = x ** exponent
    else:
        # values in range [eps**exponent, 1] where exponent >= 0
        x = x ** exponent

    result = normalize_columns(x)

    return result


class CMeans(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=8, m=2, max_iter=200, tol=1e-6, metric='euclidean', random_state=None):
        self.n_clusters = n_clusters
        self.m = m
        self.tol = tol
        self.max_iter = max_iter
        self.metric = metric
        self.random_state = random_state

    def _check_parameters(self):
        if not isinstance(self.n_clusters, int):
            raise TypeError
        if self.n_clusters <= 0:
            raise ValueError(f"n_clusters should be > 0, got {self.n_clusters} instead.")

        if not isinstance(self.m, int):
            raise TypeError
        if self.m <= 0:
            raise ValueError(f"m should be > 0, got {self.m} instead.")

        if not isinstance(self.max_iter, int):
            raise TypeError
        if self.max_iter <= 0:
            raise ValueError(f"max_iter should be > 0, got {self.max_iter} instead.")

        if not isinstance(self.tol, float):
            raise TypeError
        if self.tol <= 0:
            raise ValueError(f"tol should be > 0, got {self.tol} instead.")

        if not isinstance(self.metric, str):
            raise TypeError


    def _check_data(self, X):
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}."
            )

    def fit(self, X, init=None):
        X = self._validate_data(X, dtype=[np.float64, np.float32])
        self._check_parameters()
        self._check_data(X)
        random_state = check_random_state(self.random_state)

        data = X.T

        if init is None:
            n = data.shape[1]
            u0 = random_state.rand(self.n_clusters, n)
            u0 = normalize_columns(u0)
            init = u0.copy()
        u0 = init
        u = np.fmax(u0, np.finfo(np.float64).eps)

        # Initialize loop parameters
        jm = np.zeros(0)
        p = 0

        u2 = None
        cntr = None
        d = None

        # Main cmeans loop
        while p < self.max_iter - 1:
            u2 = u.copy()
            [cntr, u, Jjm, d] = _cmeans0(data, u2, self.n_clusters, self.m, self.metric)
            jm = np.hstack((jm, Jjm))
            p += 1

            # Stopping rule
            if np.linalg.norm(u - u2) < self.tol:
                break

        # Final calculations
        error = np.linalg.norm(u - u2)
        fpc = _fp_coeff(u)

        self.centroids_ = cntr
        self.fuzzy_labels_ = u
        self.distance_ = d
        self.objective_function_history_ = jm
        self.n_iter_ = p
        self.fuzzy_partition_coefficient_ = fpc
        self.rss_ = error

        return self

    def transform(self, X, init=None, max_iter=None, random_state=None):
        X = self._validate_data(X, dtype=[np.float64, np.float32])
        self._check_data(X)
        data = X.T
        if max_iter is None:
            max_iter = self.max_iter

        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        # Setup u0
        if init is None:
            n = data.shape[1]
            u0 = random_state.rand(self.n_clusters, n)
            u0 = normalize_columns(u0)
            init = u0.copy()
        u0 = init
        u = np.fmax(u0, np.finfo(np.float64).eps)

        # Initialize loop parameters
        jm = np.zeros(0)
        p = 0

        u2 = None
        d = None
        # Main cmeans loop
        while p < max_iter - 1:
            u2 = u.copy()
            [u, Jjm, d] = _cmeans_transform0(data, self.centroids_, u2, self.n_clusters, self.m, self.metric)
            jm = np.hstack((jm, Jjm))
            p += 1

            # Stopping rule
            if np.linalg.norm(u - u2) < self.tol:
                break

        # Final calculations
        error = np.linalg.norm(u - u2)
        fpc = _fp_coeff(u)

        fuzzy_labels_ = u
        return fuzzy_labels_.T

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.fuzzy_labels_.T

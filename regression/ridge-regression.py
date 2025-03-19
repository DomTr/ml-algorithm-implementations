import numpy as np
import numpy as np

from sklearn.metrics import mean_squared_error
import math


# Uses matrix multiplication form
def fit(X, y, lam):
    """
    This function receives training data points, then fits the ridge regression on this data
    with regularization hyperparameter lambda.
    """
    d = X.shape[1]
    weights = np.zeros((d,))
    # Closed form solution: w* = (XTX + lamI)^-1 XTy
    XT = np.transpose(X)
    XTX = XT @ X  # XTX calculated
    weights = np.linalg.solve(XTX, XT)
    assert weights.shape == (d,)
    return weights


def calculate_RMSE(X, y, w):
    rmse = 0

    pred_y = np.zeros(y.shape[0])
    for i in range(X.shape[0]):
        tmp = 0.0
        for j in range(w.shape[0]):
            tmp += w[j] * X[i][j]

        pred_y[i] = tmp

    rmse = np.sqrt(((pred_y - y) ** 2).mean())
    return rmse


def RMSE_built_in(X, y, w):
    rmse = 0
    rmse = mean_squared_error(y, X @ w)
    rmse = np.sqrt(rmse)
    return rmse

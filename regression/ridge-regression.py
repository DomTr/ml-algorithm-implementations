import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error
import math


# Uses matrix multiplication form
def fit(X, y, lam):
    """
    This function receives training data points, then fits the ridge regression on this data with regularization hyperparameter lambda.
    """
    d = X.shape[1]
    weights = np.zeros((d,))
    # Closed form solution: w* = (XTX + lamI)^-1 XTy
    M = np.transpose(X) @ X + lam * np.eye(d)
    weights = np.linalg.solve(M, np.transpose(X) @ y)
    assert weights.shape == (d,)
    return weights


def calculate_RMSE(w, X, y):
    rmse = 0
    pred_y = np.zeros(y.shape[0])
    for i in range(X.shape[0]):
        tmp = 0.0
        for j in range(w.shape[0]):
            tmp += w[j] * X[i][j]

        pred_y[i] = tmp

    rmse = np.sqrt(((pred_y - y) ** 2).mean())
    return rmse


def fitRegression(X, y):
    weights = np.zeros((y.shape[0],))
    loss_history = []
    step_size = 0.002  # learning rate
    epochs = 1000
    precision = 1e-6

    for epoch in range(epochs):
        y_pred = X @ weights  # Prediction vector computed
        error = y_pred - y  # Error vector computed
        gradient = (1 / X.shape[0]) * (X.T @ error)  # New gradient value is computed
        weights -= step_size * gradient  # Update weights

        # Compute and store loss (MSE)
        loss = (1 / (2 * X.shape[0])) * np.sum(error**2)
        loss_history.append(loss)

        # Check convergence
        if epoch > 0 and abs(loss_history[-1] - loss_history[-2]) < precision:
            print(f"Converged at epoch {epoch}")
            break

    return weights


def RMSE_built_in(X, y, w):
    rmse = 0
    rmse = mean_squared_error(y, X @ w)
    rmse = np.sqrt(rmse)
    return rmse


def average_LR_RMSE(X, y, lambdas, n_folds):
    RMSE_mat = np.zeros((n_folds, len(lambdas)))

    kf = KFold(n_splits=10)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for j, lam in enumerate(lambdas):
            w = fit(X_train, y_train, lam)
            RMSE_mat[i][j] = calculate_RMSE(w, X_test, y_test)

    avg_RMSE = np.mean(RMSE_mat, axis=0)
    return avg_RMSE


if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("regression/train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns="y")
    # print a few data samples
    # print(data.head())

    X = data.to_numpy()
    # The function calculating the average RMSE
    lambdas = [0.1, 1, 10, 100, 200]
    n_folds = 10
    avg_RMSE = average_LR_RMSE(X, y, lambdas, n_folds)
    # print(avg_RMSE)
    # Save results in the required format
    np.savetxt("regression/results.csv", avg_RMSE, fmt="%.12f")

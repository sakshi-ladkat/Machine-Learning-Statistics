import numpy as np

# ---------------------------------------------------------
# MEAN SQUARED ERROR (MSE)
# ---------------------------------------------------------
def mse(y_true, y_pred):
    """
    Mean Squared Error loss.
    """
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    """
    Derivative of MSE loss w.r.t. predictions.
    """
    return 2 * (y_pred - y_true) / y_true.size


# ---------------------------------------------------------
# BINARY CROSS ENTROPY (BCE)
# ---------------------------------------------------------
def binary_cross_entropy(y_true, y_pred):
    """
    Binary cross entropy loss (for 2-class logistic output).
    """
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_prime(y_true, y_pred):
    """
    Derivative of BCE loss w.r.t. predictions.
    """
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / y_true.size


# ---------------------------------------------------------
# MULTICLASS CROSS ENTROPY (CE)
# ---------------------------------------------------------
def cross_entropy(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred)) / y_true.size



def cross_entropy_prime(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return - (y_true / y_pred) / y_true.size






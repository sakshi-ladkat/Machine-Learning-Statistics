import numpy as np

# ---------------------------------------------------------
# BASE CLASS FOR ELEMENTWISE ACTIVATIONS
# ---------------------------------------------------------
class Activation:
    """
    Generic activation layer that applies:
        y = function(x)
    and the backward pass:
        dL/dx = dL/dy * function_prime(x)
    """
    def __init__(self, function, function_prime):
        self.function = function
        self.function_prime = function_prime

    def forward(self, input):
        self.input = input
        self.output = self.function(input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.function_prime(self.input)


# ---------------------------------------------------------
# TANH ACTIVATION
# ---------------------------------------------------------
class Tanh(Activation):
    def __init__(self):
        def f(x):
            return np.tanh(x)
        def df(x):
            t = np.tanh(x)
            return 1.0 - t * t
        super().__init__(f, df)


# ---------------------------------------------------------
# SIGMOID
# ---------------------------------------------------------
class Sigmoid(Activation):
    def __init__(self):
        def f(x):
            return 1.0 / (1.0 + np.exp(-x))
        def df(x):
            s = f(x)
            return s * (1.0 - s)
        super().__init__(f, df)


# ---------------------------------------------------------
# ReLU
# ---------------------------------------------------------
class ReLU(Activation):
    def __init__(self):
        def f(x):
            return np.maximum(0.0, x)
        def df(x):
            return (x > 0).astype(x.dtype)
        super().__init__(f, df)


# ---------------------------------------------------------
# Leaky ReLU
# ---------------------------------------------------------
class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        def f(x):
            return np.where(x > 0, x, alpha * x)
        def df(x):
            grad = np.ones_like(x)
            grad[x <= 0] = alpha
            return grad
        super().__init__(f, df)


# ---------------------------------------------------------
# ELU (Exponential Linear Unit)
# ---------------------------------------------------------
class ELU(Activation):
    def __init__(self, alpha=1.0):
        def f(x):
            return np.where(x >= 0, x, alpha * (np.exp(x) - 1.0))
        def df(x):
            return np.where(x >= 0, 1.0, alpha * np.exp(x))
        super().__init__(f, df)


# ---------------------------------------------------------
# Linear Activation (Identity)
# ---------------------------------------------------------
class Linear(Activation):
    def __init__(self):
        def f(x):
            return x
        def df(x):
            return np.ones_like(x)
        super().__init__(f, df)


# ---------------------------------------------------------
# Swish: x * sigmoid(x)
# ---------------------------------------------------------
class Swish(Activation):
    def __init__(self):
        def f(x):
            s = 1.0 / (1.0 + np.exp(-x))
            return x * s
        def df(x):
            s = 1.0 / (1.0 + np.exp(-x))
            return s + x * s * (1 - s)
        super().__init__(f, df)


# ---------------------------------------------------------
# GELU (approx)
# ---------------------------------------------------------
class GELU(Activation):
    def __init__(self):
        def f(x):
            return 0.5 * x * (1.0 + np.tanh(
                np.sqrt(2.0/np.pi) * (x + 0.044715 * x**3)
            ))
        def df(x):
            a = np.sqrt(2.0/np.pi)
            t = np.tanh(a * (x + 0.044715 * x**3))
            left = 0.5 * (1.0 + t)
            sech2 = 1.0 - t**2
            right = 0.5 * x * sech2 * a * (1 + 3 * 0.044715 * x**2)
            return left + right
        super().__init__(f, df)


# ---------------------------------------------------------
# SOFTMAX (not elementwise — handled separately)
# ---------------------------------------------------------
class Softmax:
    def __init__(self, axis=-1):
        self.axis = axis

    def forward(self, x):
        self.input = x
        x_shifted = x - np.max(x, axis=self.axis, keepdims=True)
        exp = np.exp(x_shifted)
        sum_exp = np.sum(exp, axis=self.axis, keepdims=True)
        self.output = exp / sum_exp
        return self.output

    # Fix backward to accept learning_rate
    def backward(self, dout, learning_rate=0.0):
        p = self.output

        # Single sample
        if p.ndim == 1:
            return p * (dout - np.sum(dout * p))

        # Batch case
        dot = np.sum(dout * p, axis=self.axis, keepdims=True)
        return p * (dout - dot)

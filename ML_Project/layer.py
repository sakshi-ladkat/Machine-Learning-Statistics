import numpy as np
from scipy import signal

# -------------------------------
# Base Layer
# -------------------------------
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError("Forward method not implemented")

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError("Backward method not implemented")


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.bias = np.zeros((output_size,))   # 1D bias

    def forward(self, input):
        self.input = input.reshape(-1)  # ensure 1D
        self.output = np.dot(self.weights, self.input) + self.bias
        return self.output

    def backward(self, output_gradient, learning_rate):
        output_gradient = output_gradient.reshape(-1)  # ensure 1D

        # Gradients
        weight_gradient = np.outer(output_gradient, self.input)  # (output_size, input_size)
        input_gradient = np.dot(self.weights.T, output_gradient) # (input_size,)

        # Update parameters
        self.weights -= learning_rate * weight_gradient
        self.bias -= learning_rate * output_gradient  # 1D update

        return input_gradient




# -------------------------------
# Convolutional Layer
# -------------------------------
class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        # input_shape = (channels, height, width)
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth

        self.output_shape = (
            depth,
            input_height - kernel_size + 1,
            input_width - kernel_size + 1
        )

        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        limit = np.sqrt(1 / (input_depth * kernel_size * kernel_size))
        self.kernels = np.random.uniform(-limit, limit, self.kernels_shape)
        self.biases = np.zeros(self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)

        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(
                    self.input[j],
                    self.kernels[i, j],
                    mode="valid"
                )
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(
                    self.input[j],
                    output_gradient[i],
                    mode="valid"
                )
                input_gradient[j] += signal.convolve2d(
                    output_gradient[i],
                    np.flip(self.kernels[i, j], axis=(0, 1)),
                    mode="full"
                )

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient

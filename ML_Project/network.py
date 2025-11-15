import numpy as np

# ---------------------------------------------------------
# PREDICT FUNCTION
# ---------------------------------------------------------
def predict(network, input):
    """
    Forward pass through the network to get predictions.

    Parameters:
        network : list of layers
        input   : input sample, shape depends on first layer

    Returns:
        output  : network output
    """
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


# ---------------------------------------------------------
# TRAIN FUNCTION
# ---------------------------------------------------------
def train(network, loss, loss_prime, x_train, y_train, epochs=1000, learning_rate=0.01, verbose=True):
    """
    Train the network using vanilla gradient descent.

    Parameters:
        network       : list of layers (Dense, Convolutional, Activation, etc.)
        loss          : loss function (e.g., mse, cross_entropy)
        loss_prime    : derivative of loss function
        x_train       : training data, shape (num_samples, ...)
        y_train       : training labels, shape (num_samples, ...)
        epochs        : number of training iterations
        learning_rate : learning rate for weight updates
        verbose       : if True, prints loss per epoch

    Returns:
        epoch_losses : list of average loss values per epoch (for plotting)
    """
    epoch_losses = []  # store loss per epoch

    for e in range(epochs):
        error = 0  # cumulative error for the epoch

        # Iterate over each training sample
        for x, y in zip(x_train, y_train):

            # -------------------------
            # FORWARD PASS
            # -------------------------
            output = predict(network, x)

            # -------------------------
            # ACCUMULATE LOSS
            # -------------------------
            error += loss(y, output)

            # -------------------------
            # BACKWARD PASS
            # -------------------------
            grad = loss_prime(y, output)  # gradient of loss w.r.t. output
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        # Average error for the epoch
        error /= len(x_train)
        epoch_losses.append(error)

        # Print epoch summary
        if verbose:
            print(f"Epoch {e+1}/{epochs}, Loss={error:.6f}")

    return epoch_losses

                

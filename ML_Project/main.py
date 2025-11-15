import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical

from layer import Dense
from activations import ReLU, LeakyReLU, Tanh, ELU, Swish, GELU, Softmax
from losses import cross_entropy, cross_entropy_prime

# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------
def preprocess(x, y, limit):
    x = x.reshape(len(x), 28*28).astype("float32") / 255.0   # Flatten to (num_samples, 784)
    y = to_categorical(y)                                    # Shape (num_samples, 10)
    return x[:limit], y[:limit]

# ---------------------------------------------------------
# PREDICT FUNCTION
# ---------------------------------------------------------
def predict(network, input_sample):
    output = input_sample.reshape(-1)   # ensure 1D input
    for layer in network:
        output = layer.forward(output)
    return output.reshape(-1)           # flatten output to 1D

# ---------------------------------------------------------
# TRAIN FUNCTION
# ---------------------------------------------------------
def train(network, loss, loss_prime, x_train, y_train, epochs=100, learning_rate=0.01, verbose=True):
    epoch_losses = []

    for e in range(epochs):
        error = 0

        for x, y in zip(x_train, y_train):
            # Forward
            output = predict(network, x)
            y_flat = y.reshape(-1)

            # Loss accumulation
            error += loss(y_flat, output)

            # Backward
            grad = loss_prime(y_flat, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)
        epoch_losses.append(error)

        if verbose:
            print(f"Epoch {e+1}/{epochs}, Loss={error:.6f}")

    return epoch_losses

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess(x_train, y_train, 2000)
x_test, y_test = preprocess(x_test, y_test, 500)

# ---------------------------------------------------------
# Hidden activation set
# ---------------------------------------------------------
hidden_activation_set = [
    ("ReLU",      ReLU),
    ("LeakyReLU", LeakyReLU),
    ("Tanh",      Tanh),
    ("ELU",       ELU),
    ("Swish",     Swish),
    ("GELU",      GELU)
]

# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------
loss_curves = {}
accuracy_results = {}

for name, Act in hidden_activation_set:
    print("\n============================================")
    print(f" Training with Hidden Activation: {name}")
    print("============================================")

    # Build network
    network = [
        Dense(28*28, 64),
        Act(),
        Dense(64, 10),
        Softmax()  # Always softmax output for classification
    ]

    # Train network
    losses = train(
        network,
        cross_entropy,
        cross_entropy_prime,
        x_train,
        y_train,
        epochs=100,         # Reduced for faster testing
        learning_rate=0.001
    )

    loss_curves[name] = losses

    # Test accuracy
    correct = 0
    total = len(x_test)

    for x, y in zip(x_test, y_test):
        out = predict(network, x)
        label = np.argmax(y)     # y is one-hot
        pred = np.argmax(out)
        if pred == label:
            correct += 1

    accuracy = correct / total
    accuracy_results[name] = accuracy
    print(f"\n>>> {name} Accuracy = {accuracy:.4f}")

# ---------------------------------------------------------
# Plot loss curves
# ---------------------------------------------------------
plt.figure(figsize=(10,6))
for name, losses in loss_curves.items():
    plt.plot(losses, label=name)

plt.title("Training Loss vs Epochs for Different Hidden Activations")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------------------------------------
# Final accuracy summary
# ---------------------------------------------------------
print("\n\n============== FINAL ACCURACY RESULTS ==============")
for name, acc in accuracy_results.items():
    print(f"{name:10s} : {acc:.4f}")

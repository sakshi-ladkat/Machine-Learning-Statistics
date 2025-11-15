import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from layer import Dense,Convolutional
from reshape import Reshape
from activations import ReLU, LeakyReLU, Tanh, ELU, Swish, GELU, Softmax
from losses import cross_entropy, cross_entropy_prime
from network import train, predict


# -----------------------------
# --- Data preprocessing ---
# -----------------------------
def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]

    all_indices = np.hstack((zero_index, one_index))
    np.random.shuffle(all_indices)

    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28).astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y


# -----------------------------
# --- Hidden activation set ---
# -----------------------------
hidden_activation_set = [
    ("ReLU",       ReLU),
    ("LeakyReLU",  LeakyReLU),
    ("Tanh",       Tanh),
    ("ELU",        ELU),
    ("Swish",      Swish),
    ("GELU",       GELU)
]


# -----------------------------
# --- Load MNIST ---
# -----------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)   # small for speed
x_test, y_test = preprocess_data(x_test, y_test, 200)


# -----------------------------
# --- Train and evaluate ---
# -----------------------------
loss_curves = {}
accuracy_results = {}

for name, Act in hidden_activation_set:

    print("\n============================================")
    print(f" Training with Hidden Activation: {name}")
    print("============================================")

    network = [
        Convolutional((1, 28, 28), 3, 5),
        Act(),                             # choose hidden activation
        Reshape((5, 26, 26), (5*26*26, 1)),
        Dense(5*26*26, 100),
        Act(),                             # hidden activation after dense
        Dense(100, 2),
        Softmax()                          # always softmax at output
    ]

    # Train and collect loss per epoch
    losses = train(
        network,
        cross_entropy,
        cross_entropy_prime,
        x_train,
        y_train,
        epochs=20,
        learning_rate=0.1
    )
    loss_curves[name] = losses

    # Evaluate accuracy
    correct = 0
    for x, y in zip(x_test, y_test):
        out = predict(network, x)
        if np.argmax(out) == np.argmax(y):
            correct += 1

    accuracy = correct / len(x_test)
    accuracy_results[name] = accuracy
    print(f"\n>>> {name} Accuracy = {accuracy:.4f}")


# -----------------------------
# --- Plot loss curves ---
# -----------------------------
plt.figure(figsize=(10,6))
for name, losses in loss_curves.items():
    plt.plot(losses, label=name)
plt.title("Training Loss vs Epochs for Different Hidden Activations (CNN)")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.grid(True)
plt.show()


# -----------------------------
# --- Final accuracy summary ---
# -----------------------------
print("\n\n============== FINAL ACCURACY RESULTS ==============")
for name, acc in accuracy_results.items():
    print(f"{name:10s} : {acc:.4f}")

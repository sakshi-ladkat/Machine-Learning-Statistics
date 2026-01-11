import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load CSV file manually
file_path = "iris.csv"
with open(file_path, 'r') as file:
    lines = file.readlines()

# Strip newline and split by comma
data = [line.strip().split(',') for line in lines]

# Classes of Classification
species = [row[4] for row in data[1:]]  # Skip header

# One Hot Encoding
unique_species = list(set(species))
one_hot = [[1 if specie == uc else 0 for uc in unique_species] for specie in species]
one_hot = np.array(one_hot, dtype=float)

# Convert original data to NumPy
data_np = np.array(data[1:], dtype=object)
rows, cols = data_np.shape

# Combine features + labels
data_np1 = []
for i in range(rows):
    row = list(data_np[i][:4]) + list(one_hot[i])
    data_np1.append(row)

data_np1 = np.array(data_np1, dtype=float)

# Split features and labels
X = data_np1[:, :4]
y = data_np1[:, 4:]

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, one_hot, test_size=0.2, random_state=42)

# Initialize weights and bias
num_features = X_train.shape[1]
num_classes = y_train.shape[1]
W = np.zeros((num_features, num_classes))
b = np.zeros((1, num_classes))

# Softmax function
def softmax(z):
    exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Cross entropy loss
def cross_entropy(y_true, y_pred):
    n = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-15)) / n

# Training parameters
learning_rate = 0.1
epochs = 500
losses = []

# Training loop
for epoch in range(epochs):

    # Forward pass
    z = np.dot(X_train, W) + b
    y_pred = softmax(z)

    # Loss calculation
    loss = cross_entropy(y_train, y_pred)
    losses.append(loss)

    # Backward pass
    dz = y_pred - y_train
    dW = np.dot(X_train.T, dz) / X_train.shape[0]
    db = np.sum(dz, axis=0, keepdims=True) / X_train.shape[0]

    # Gradient descent update
    W -= learning_rate * dW
    b -= learning_rate * db

    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} => Loss: {loss:.4f}")

# Testing and predictions after training 
z_test = np.dot(X_test, W) + b
y_pred_test = softmax(z_test)
y_pred_labels = np.argmax(y_pred_test, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Confusion Matrix after training 
conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
for true, pred in zip(y_true_labels, y_pred_labels):
    conf_matrix[true][pred] += 1

print("\nConfusion Matrix:")
print(conf_matrix)

# Plot loss curve AFTER training 
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), losses, marker='o', color='blue')
plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")
plt.title("Loss vs Epochs")
plt.grid()
plt.show()

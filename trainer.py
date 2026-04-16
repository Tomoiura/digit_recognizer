"""Training loop for CNN on MNIST — SGD with mini-batches."""

import numpy as np
from model import CNN, cross_entropy_loss, softmax


def train(model: CNN, images: np.ndarray, labels: np.ndarray,
          epochs: int = 10, batch_size: int = 64, lr: float = 0.01,
          val_images: np.ndarray = None, val_labels: np.ndarray = None) -> dict:
    """
    Train the CNN with SGD.

    Args:
        model: CNN instance
        images: (N, 28, 28) float32
        labels: (N,) uint8
        epochs: number of epochs
        batch_size: mini-batch size
        lr: learning rate

    Returns:
        dict with training history
    """
    N = images.shape[0]
    # Reshape for conv: (N, 1, 28, 28)
    X = images[:, None, :, :]
    y = labels.astype(np.int64)

    history = {"loss": [], "acc": [], "val_acc": []}

    for epoch in range(epochs):
        # Shuffle
        perm = np.random.permutation(N)
        X_shuf = X[perm]
        y_shuf = y[perm]

        epoch_loss = 0.0
        epoch_correct = 0
        n_batches = 0

        for i in range(0, N, batch_size):
            xb = X_shuf[i:i+batch_size]
            yb = y_shuf[i:i+batch_size]

            # Forward
            logits = model.forward(xb)
            loss, dlogits = cross_entropy_loss(logits, yb)

            # Backward
            grads = model.backward(dlogits)

            # SGD update
            for key in model.params:
                model.params[key] -= lr * grads[key]

            epoch_loss += loss * xb.shape[0]
            preds = logits.argmax(axis=1)
            epoch_correct += (preds == yb).sum()
            n_batches += 1

        avg_loss = epoch_loss / N
        avg_acc = epoch_correct / N
        history["loss"].append(float(avg_loss))
        history["acc"].append(float(avg_acc))

        # Validation
        val_acc_str = ""
        if val_images is not None:
            va = evaluate(model, val_images, val_labels)
            history["val_acc"].append(va)
            val_acc_str = f"  val_acc={va:.4f}"

        print(f"Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}  "
              f"acc={avg_acc:.4f}{val_acc_str}")

    return history


def evaluate(model: CNN, images: np.ndarray, labels: np.ndarray,
             batch_size: int = 256) -> float:
    X = images[:, None, :, :]
    y = labels.astype(np.int64)
    correct = 0
    for i in range(0, len(X), batch_size):
        xb = X[i:i+batch_size]
        yb = y[i:i+batch_size]
        logits = model.forward(xb)
        correct += (logits.argmax(axis=1) == yb).sum()
    return float(correct / len(X))

"""CNN model for MNIST digit recognition — pure NumPy, forward + backward."""

import numpy as np


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _he_init(fan_in: int, shape: tuple) -> np.ndarray:
    return np.random.randn(*shape).astype(np.float32) * np.sqrt(2.0 / fan_in)


# ---------------------------------------------------------------------------
# Layers (forward / backward)
# ---------------------------------------------------------------------------

def conv2d_forward(x, W, b):
    """
    x: (N, C_in, H, W)
    W: (C_out, C_in, kH, kW)
    b: (C_out,)
    returns: out (N, C_out, H', W'), cache
    """
    N, C_in, H, Wid = x.shape
    C_out, _, kH, kW = W.shape
    H_out = H - kH + 1
    W_out = Wid - kW + 1

    # im2col
    cols = np.zeros((N, C_in, kH, kW, H_out, W_out), dtype=x.dtype)
    for i in range(kH):
        for j in range(kW):
            cols[:, :, i, j, :, :] = x[:, :, i:i+H_out, j:j+W_out]
    cols = cols.reshape(N, C_in * kH * kW, H_out * W_out)  # (N, CkkH*kW, H'*W')

    W_flat = W.reshape(C_out, -1)  # (C_out, C_in*kH*kW)
    out = W_flat @ cols  # (N, C_out, H'*W')  — broadcast over N via matmul
    # Actually need to handle batch: use einsum or loop-free
    # np.matmul broadcasts: (C_out, C_in*kH*kW) @ (N, C_in*kH*kW, H'*W') won't work
    # Transpose cols: (N, C_in*kH*kW, H'W') -> ok with einsum
    out = np.einsum("ck,nkp->ncp", W_flat, cols)  # (N, C_out, H'*W')
    out = out.reshape(N, C_out, H_out, W_out)
    out += b[None, :, None, None]

    cache = (x, W, cols, H_out, W_out)
    return out, cache


def conv2d_backward(dout, cache):
    x, W, cols, H_out, W_out = cache
    N, C_in, H, Wid = x.shape
    C_out, _, kH, kW = W.shape

    dout_flat = dout.reshape(N, C_out, H_out * W_out)  # (N, C_out, H'W')
    W_flat = W.reshape(C_out, -1)

    # dW
    # cols: (N, C_in*kH*kW, H'W')
    dW_flat = np.einsum("ncp,nkp->ck", dout_flat, cols)  # (C_out, C_in*kH*kW)
    dW = dW_flat.reshape(W.shape)

    # db
    db = dout.sum(axis=(0, 2, 3))

    # dx via col2im
    dcols = np.einsum("ck,ncp->nkp", W_flat, dout_flat)  # (N, C_in*kH*kW, H'W')
    dcols = dcols.reshape(N, C_in, kH, kW, H_out, W_out)
    dx = np.zeros_like(x)
    for i in range(kH):
        for j in range(kW):
            dx[:, :, i:i+H_out, j:j+W_out] += dcols[:, :, i, j, :, :]

    return dx, dW, db


def relu_forward(x):
    mask = x > 0
    return x * mask, mask


def relu_backward(dout, mask):
    return dout * mask


def maxpool2d_forward(x, size=2):
    N, C, H, W = x.shape
    H2, W2 = H // size, W // size
    x_reshaped = x.reshape(N, C, H2, size, W2, size)
    out = x_reshaped.max(axis=(3, 5))
    # Store argmax for backward
    mask = (x_reshaped == out[:, :, :, None, :, None])
    cache = (x.shape, mask, size)
    return out, cache


def maxpool2d_backward(dout, cache):
    x_shape, mask, size = cache
    N, C, H2, W2 = dout.shape
    dout_expanded = dout[:, :, :, None, :, None]  # (N,C,H2,1,W2,1)
    # Distribute gradient equally among max positions (usually just one)
    count = mask.sum(axis=(3, 5), keepdims=True).clip(min=1)
    dx = (mask * dout_expanded / count).reshape(x_shape)
    return dx


def fc_forward(x, W, b):
    """x: (N, D), W: (D, M), b: (M,)"""
    out = x @ W + b
    cache = (x, W)
    return out, cache


def fc_backward(dout, cache):
    x, W = cache
    dx = dout @ W.T
    dW = x.T @ dout
    db = dout.sum(axis=0)
    return dx, dW, db


def softmax(logits):
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy_loss(logits, labels):
    N = logits.shape[0]
    probs = softmax(logits)
    log_probs = np.log(probs.clip(min=1e-12))
    loss = -log_probs[np.arange(N), labels].mean()
    dlogits = probs.copy()
    dlogits[np.arange(N), labels] -= 1.0
    dlogits /= N
    return loss, dlogits


# ---------------------------------------------------------------------------
# CNN Model
# ---------------------------------------------------------------------------

class CNN:
    """
    Small CNN for MNIST:
        Conv1(1→8, 5×5) → ReLU → MaxPool(2)   → 8×12×12
        Conv2(8→16, 3×3) → ReLU → MaxPool(2)   → 16×5×5
        Flatten(400) → FC1(400→64) → ReLU → FC2(64→10)
    """

    def __init__(self):
        self.params = {
            "conv1_W": _he_init(1*5*5, (8, 1, 5, 5)),
            "conv1_b": np.zeros(8, dtype=np.float32),
            "conv2_W": _he_init(8*3*3, (16, 8, 3, 3)),
            "conv2_b": np.zeros(16, dtype=np.float32),
            "fc1_W": _he_init(400, (400, 64)),
            "fc1_b": np.zeros(64, dtype=np.float32),
            "fc2_W": _he_init(64, (64, 10)),
            "fc2_b": np.zeros(10, dtype=np.float32),
        }

    def forward(self, x, return_intermediates=False):
        """
        x: (N, 1, 28, 28)
        returns: logits (N, 10), optionally intermediate activations dict
        """
        p = self.params
        intermediates = {}

        # Conv1 → ReLU → Pool
        h, c_conv1 = conv2d_forward(x, p["conv1_W"], p["conv1_b"])
        intermediates["conv1_pre"] = h.copy()
        h, c_relu1 = relu_forward(h)
        intermediates["conv1"] = h.copy()  # (N, 8, 24, 24)
        h, c_pool1 = maxpool2d_forward(h, 2)
        intermediates["pool1"] = h.copy()  # (N, 8, 12, 12)

        # Conv2 → ReLU → Pool
        h, c_conv2 = conv2d_forward(h, p["conv2_W"], p["conv2_b"])
        intermediates["conv2_pre"] = h.copy()
        h, c_relu2 = relu_forward(h)
        intermediates["conv2"] = h.copy()  # (N, 16, 10, 10)
        h, c_pool2 = maxpool2d_forward(h, 2)
        intermediates["pool2"] = h.copy()  # (N, 16, 5, 5)

        # Flatten
        N = x.shape[0]
        h_flat = h.reshape(N, -1)  # (N, 400)

        # FC1 → ReLU
        h, c_fc1 = fc_forward(h_flat, p["fc1_W"], p["fc1_b"])
        intermediates["fc1_pre"] = h.copy()
        h, c_relu3 = relu_forward(h)
        intermediates["fc1"] = h.copy()  # (N, 64)

        # FC2
        logits, c_fc2 = fc_forward(h, p["fc2_W"], p["fc2_b"])
        intermediates["logits"] = logits.copy()  # (N, 10)

        self._cache = (c_conv1, c_relu1, c_pool1,
                       c_conv2, c_relu2, c_pool2,
                       h_flat.shape, c_fc1, c_relu3, c_fc2)

        if return_intermediates:
            return logits, intermediates
        return logits

    def backward(self, dlogits):
        (c_conv1, c_relu1, c_pool1,
         c_conv2, c_relu2, c_pool2,
         flat_shape, c_fc1, c_relu3, c_fc2) = self._cache
        grads = {}

        # FC2
        dh, grads["fc2_W"], grads["fc2_b"] = fc_backward(dlogits, c_fc2)

        # ReLU3
        dh = relu_backward(dh, c_relu3)

        # FC1
        dh, grads["fc1_W"], grads["fc1_b"] = fc_backward(dh, c_fc1)

        # Unflatten
        dh = dh.reshape(flat_shape[0], 16, 5, 5)

        # Pool2
        dh = maxpool2d_backward(dh, c_pool2)

        # ReLU2
        dh = relu_backward(dh, c_relu2)

        # Conv2
        dh, grads["conv2_W"], grads["conv2_b"] = conv2d_backward(dh, c_conv2)

        # Pool1
        dh = maxpool2d_backward(dh, c_pool1)

        # ReLU1
        dh = relu_backward(dh, c_relu1)

        # Conv1
        _, grads["conv1_W"], grads["conv1_b"] = conv2d_backward(dh, c_conv1)

        return grads

    def get_weights_for_js(self) -> dict:
        """Export weights as plain lists for JSON embedding."""
        return {k: v.tolist() for k, v in self.params.items()}

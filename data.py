"""MNIST data download and loading."""

import gzip
import hashlib
import os
import struct
import urllib.request

import numpy as np

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"
FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    "test_images":  ("t10k-images-idx3-ubyte.gz",  "9fb629c4189551a2d022fa330f9573f3"),
    "test_labels":  ("t10k-labels-idx1-ubyte.gz",  "ec29112dd5afa0611ce80d1b7f02629c"),
}


def _download(dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    for name, (fname, md5) in FILES.items():
        path = os.path.join(dest_dir, fname)
        if os.path.exists(path):
            continue
        url = f"{MIRROR}/{fname}"
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, path)
        h = hashlib.md5(open(path, "rb").read()).hexdigest()
        if h != md5:
            raise ValueError(f"MD5 mismatch for {fname}: {h} != {md5}")


def _read_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols).astype(np.float32) / 255.0


def _read_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_mnist(data_dir: str = "data") -> dict:
    """Download MNIST and return train/test splits.

    Returns dict with keys:
        train_images: (60000, 28, 28) float32 [0,1]
        train_labels: (60000,) uint8
        test_images:  (10000, 28, 28) float32 [0,1]
        test_labels:  (10000,) uint8
    """
    _download(data_dir)
    return {
        "train_images": _read_images(os.path.join(data_dir, FILES["train_images"][0])),
        "train_labels": _read_labels(os.path.join(data_dir, FILES["train_labels"][0])),
        "test_images":  _read_images(os.path.join(data_dir, FILES["test_images"][0])),
        "test_labels":  _read_labels(os.path.join(data_dir, FILES["test_labels"][0])),
    }

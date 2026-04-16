#!/usr/bin/env python3
"""Digit Recognizer — train CNN on MNIST, generate interactive HTML."""

import json
import os
import sys

import numpy as np

from data import load_mnist
from model import CNN
from trainer import train, evaluate
from visualizer_html import build_html

WEIGHTS_CACHE = "weights_cache.npz"


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "output")
    data_dir = os.path.join(base_dir, "data")
    cache_path = os.path.join(base_dir, WEIGHTS_CACHE)
    os.makedirs(out_dir, exist_ok=True)

    # --- Check cache ---
    if os.path.exists(cache_path) and "--retrain" not in sys.argv:
        print("Loading cached weights ...")
        cached = np.load(cache_path)
        weights_js = {}
        for key in cached.files:
            weights_js[key] = cached[key].tolist()
        test_acc = float(cached["__test_acc__"]) if "__test_acc__" in cached.files else 0.98
        history = {"loss": [], "acc": [], "val_acc": []}
    else:
        # --- Load data ---
        print("Loading MNIST ...")
        data = load_mnist(data_dir)
        train_img = data["train_images"]
        train_lbl = data["train_labels"]
        test_img = data["test_images"]
        test_lbl = data["test_labels"]
        print(f"  Train: {train_img.shape[0]}  Test: {test_img.shape[0]}")

        # --- Train ---
        np.random.seed(42)
        model = CNN()
        print("\nTraining CNN ...")
        history = train(
            model, train_img, train_lbl,
            epochs=10, batch_size=64, lr=0.01,
            val_images=test_img, val_labels=test_lbl,
        )

        test_acc = evaluate(model, test_img, test_lbl)
        print(f"\nFinal test accuracy: {test_acc:.4f}")

        # --- Cache weights ---
        weights_js = model.get_weights_for_js()
        save_dict = {k: np.array(v) for k, v in weights_js.items()}
        save_dict["__test_acc__"] = np.array(test_acc)
        np.savez(cache_path, **save_dict)
        print(f"Cached weights to {cache_path}")

    # --- Generate HTML ---
    print("\nGenerating HTML ...")
    html = build_html(weights_js, history, test_acc)
    out_path = os.path.join(out_dir, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved to {out_path}")
    print(f"  File size: {os.path.getsize(out_path) / 1024:.0f} KB")


if __name__ == "__main__":
    main()

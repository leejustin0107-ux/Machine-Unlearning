import time
from typing import Dict, Tuple
import torch
from torch import nn, optim
from tqdm import tqdm

from src.evaluation.metrics import evaluate_model


def train_model(model, train_loader, device, epochs: int = 3, lr: float = 1e-3) -> Tuple[object, float]:
    """Generic supervised training loop."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start = time.time()

    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=float(loss.item()))

    runtime = time.time() - start
    return model, runtime


def train_baseline(model, train_loader, test_loader, device, epochs: int = 3, lr: float = 1e-3) -> Dict[str, float]:
    model, runtime = train_model(model, train_loader, device, epochs=epochs, lr=lr)
    test_metrics = evaluate_model(model, test_loader, device)
    return {
        "runtime": runtime,
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
    }

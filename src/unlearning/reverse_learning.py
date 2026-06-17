import time
from itertools import cycle
from typing import Dict
import torch
from torch import nn, optim

from src.evaluation.metrics import evaluate_model


def reverse_learning_unlearn(
    model,
    forget_loader,
    retain_loader,
    test_loader,
    device,
    epochs: int = 3,
    lr: float = 1e-4,
    alpha: float = 0.5,
) -> Dict[str, object]:
    """
    Reverse learning / gradient ascent unlearning.

    The loss minimizes retain loss while maximizing forget loss:
        total_loss = retain_loss - alpha * forget_loss

    Tune alpha and epochs carefully to avoid damaging retain performance.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start = time.time()

    retain_cycle = cycle(retain_loader)

    for _ in range(epochs):
        model.train()
        for forget_x, forget_y in forget_loader:
            retain_x, retain_y = next(retain_cycle)
            forget_x, forget_y = forget_x.to(device), forget_y.to(device)
            retain_x, retain_y = retain_x.to(device), retain_y.to(device)

            optimizer.zero_grad()
            forget_loss = criterion(model(forget_x), forget_y)
            retain_loss = criterion(model(retain_x), retain_y)
            total_loss = retain_loss - alpha * forget_loss
            total_loss.backward()
            optimizer.step()

    runtime = time.time() - start
    return {
        "model": model,
        "runtime": runtime,
        "test": evaluate_model(model, test_loader, device),
        "retain": evaluate_model(model, retain_loader, device),
        "forget": evaluate_model(model, forget_loader, device),
    }

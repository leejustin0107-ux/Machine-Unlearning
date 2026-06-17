import time
from itertools import cycle
from typing import Dict
import torch
from torch import nn, optim

from src.evaluation.metrics import evaluate_model


def _random_wrong_labels(labels, num_classes: int = 10):
    random_labels = torch.randint(0, num_classes, labels.shape, device=labels.device)
    same = random_labels == labels
    random_labels[same] = (random_labels[same] + 1) % num_classes
    return random_labels


def random_label_unlearn(
    model,
    forget_loader,
    retain_loader,
    test_loader,
    device,
    epochs: int = 2,
    lr: float = 1e-4,
    retain_weight: float = 1.0,
) -> Dict[str, object]:
    """
    Random-label unlearning baseline.

    The model is trained to map forget samples to incorrect labels while still anchoring on retain data.
    This is a simple comparison method and should be validated carefully.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    retain_cycle = cycle(retain_loader)
    start = time.time()

    for _ in range(epochs):
        model.train()
        for forget_x, forget_y in forget_loader:
            retain_x, retain_y = next(retain_cycle)
            forget_x, forget_y = forget_x.to(device), forget_y.to(device)
            retain_x, retain_y = retain_x.to(device), retain_y.to(device)
            wrong_y = _random_wrong_labels(forget_y)

            optimizer.zero_grad()
            forget_loss = criterion(model(forget_x), wrong_y)
            retain_loss = criterion(model(retain_x), retain_y)
            loss = forget_loss + retain_weight * retain_loss
            loss.backward()
            optimizer.step()

    runtime = time.time() - start
    return {
        "model": model,
        "runtime": runtime,
        "test": evaluate_model(model, test_loader, device),
        "retain": evaluate_model(model, retain_loader, device),
        "forget": evaluate_model(model, forget_loader, device),
    }

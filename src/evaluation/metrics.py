from typing import Dict
import torch
from torch import nn


@torch.no_grad()
def evaluate_model(model, data_loader, device) -> Dict[str, float]:
    """Return loss and accuracy for a model on a given data loader."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        preds = logits.argmax(dim=1)
        total_loss += loss.item()
        correct += (preds == y).sum().item()
        total += y.size(0)

    if total == 0:
        return {"loss": 0.0, "accuracy": 0.0, "total": 0}

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "total": total,
    }

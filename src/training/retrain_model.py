from typing import Dict
from src.models.cnn_model import SimpleCNN
from src.training.train_baseline import train_model
from src.evaluation.metrics import evaluate_model


def full_retrain(retain_loader, test_loader, forget_loader, device, epochs: int = 3, lr: float = 1e-3) -> Dict[str, object]:
    """Train a fresh CNN from scratch using only retained data."""
    model = SimpleCNN()
    model, runtime = train_model(model, retain_loader, device, epochs=epochs, lr=lr)
    return {
        "model": model,
        "runtime": runtime,
        "test": evaluate_model(model, test_loader, device),
        "retain": evaluate_model(model, retain_loader, device),
        "forget": evaluate_model(model, forget_loader, device),
    }

from typing import Dict
from src.training.train_baseline import train_model
from src.evaluation.metrics import evaluate_model


def retain_finetune_unlearn(model, retain_loader, forget_loader, test_loader, device, epochs: int = 2, lr: float = 1e-4) -> Dict[str, object]:
    """Simple baseline: fine-tune the existing model using retained data only."""
    model, runtime = train_model(model, retain_loader, device, epochs=epochs, lr=lr)
    return {
        "model": model,
        "runtime": runtime,
        "test": evaluate_model(model, test_loader, device),
        "retain": evaluate_model(model, retain_loader, device),
        "forget": evaluate_model(model, forget_loader, device),
    }

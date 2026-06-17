from typing import Dict
import torch
import torch.nn.functional as F


@torch.no_grad()
def confidence_on_loader(model, data_loader, device) -> Dict[str, float]:
    """
    Simple confidence-based forgetting verification.

    Lower average confidence on forget samples after unlearning may suggest reduced influence,
    but this is not a formal privacy guarantee.
    """
    model.eval()
    confidences = []
    for x, _ in data_loader:
        x = x.to(device)
        probs = F.softmax(model(x), dim=1)
        max_conf = probs.max(dim=1).values
        confidences.extend(max_conf.detach().cpu().tolist())

    if not confidences:
        return {"avg_confidence": 0.0}
    return {"avg_confidence": sum(confidences) / len(confidences)}

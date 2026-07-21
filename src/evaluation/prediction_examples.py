from typing import List, Dict
import torch
import torch.nn.functional as F


@torch.no_grad()
def get_prediction_examples(
    baseline_model,
    unlearned_model,
    data_loader,
    device,
    max_samples: int = 6,
) -> List[Dict]:
    """
    Compare baseline and unlearned model predictions on forget samples.

    Returns example images with:
    - true label
    - baseline prediction and confidence
    - unlearned prediction and confidence
    """

    baseline_model.eval()
    unlearned_model.eval()

    examples = []

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        baseline_logits = baseline_model(x)
        unlearned_logits = unlearned_model(x)

        baseline_probs = F.softmax(baseline_logits, dim=1)
        unlearned_probs = F.softmax(unlearned_logits, dim=1)

        baseline_conf, baseline_pred = baseline_probs.max(dim=1)
        unlearned_conf, unlearned_pred = unlearned_probs.max(dim=1)

        for i in range(x.size(0)):
            # Unnormalize MNIST image for display
            image = x[i].detach().cpu().squeeze()
            image = image * 0.3081 + 0.1307
            image = image.clamp(0, 1).numpy()

            examples.append(
                {
                    "image": image,
                    "true_label": int(y[i].cpu().item()),
                    "baseline_pred": int(baseline_pred[i].cpu().item()),
                    "baseline_conf": float(baseline_conf[i].cpu().item()),
                    "unlearned_pred": int(unlearned_pred[i].cpu().item()),
                    "unlearned_conf": float(unlearned_conf[i].cpu().item()),
                }
            )

            if len(examples) >= max_samples:
                return examples

    return examples
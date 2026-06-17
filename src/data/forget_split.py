from typing import Tuple, Optional
import random
import torch
from torch.utils.data import Subset


def create_forget_retain_split(
    dataset,
    forget_digit: Optional[int] = None,
    forget_size: int = 500,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    """
    Create forget and retain subsets from MNIST.

    If forget_digit is provided, samples are selected only from that digit.
    If forget_size is smaller than the available target samples, a random subset is used.
    """
    rng = random.Random(seed)

    targets = dataset.targets
    if isinstance(targets, torch.Tensor):
        targets = targets.tolist()

    if forget_digit is None:
        candidate_indices = list(range(len(dataset)))
    else:
        candidate_indices = [idx for idx, label in enumerate(targets) if int(label) == int(forget_digit)]

    rng.shuffle(candidate_indices)
    forget_indices = set(candidate_indices[: min(forget_size, len(candidate_indices))])
    retain_indices = [idx for idx in range(len(dataset)) if idx not in forget_indices]

    forget_subset = Subset(dataset, sorted(forget_indices))
    retain_subset = Subset(dataset, retain_indices)
    return forget_subset, retain_subset

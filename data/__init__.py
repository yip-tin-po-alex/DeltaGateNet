from .sadt import load_sadt
from .seed_vig import SEED_VIG_FILES, load_seed_vig
from .splits import split_indices

DATASET_DEFAULTS = {
    "seed-vig": {"num_channels": 17, "num_classes": 3},
    "sadt": {"num_channels": 30, "num_classes": 2},
}

__all__ = [
    "DATASET_DEFAULTS",
    "SEED_VIG_FILES",
    "load_sadt",
    "load_seed_vig",
    "split_indices",
]

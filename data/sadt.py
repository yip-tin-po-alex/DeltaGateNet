import glob
import os

import numpy as np
from scipy.io import loadmat


def _find_mat_file(data_dir):
    if os.path.isfile(data_dir) and data_dir.endswith(".mat"):
        return data_dir

    mat_files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(
            f"No .mat file found in {data_dir}. "
            "Download the SADT Figshare release and place it under datasets/. "
            "See README.md for the official links and directory layout."
        )
    return mat_files[0]


def load_sadt(data_dir, num_channels):
    """
    Load SADT from a single .mat containing EEGsample, subindex, and substate.

    Works for both the balanced (2022) and unbalanced (2952) releases.
    Returns EEG as (N, C, T).
    """
    mat_path = _find_mat_file(data_dir)
    mat = loadmat(mat_path)

    for key in ("EEGsample", "subindex", "substate"):
        if key not in mat:
            raise KeyError(
                f"{mat_path} is missing '{key}'. "
                "The SADT Figshare files must keep EEGsample, subindex, and substate."
            )

    eeg = np.array(mat["EEGsample"])
    if eeg.ndim != 3:
        raise ValueError(f"EEGsample must be 3-D (N, C, T), got shape {eeg.shape}")

    if eeg.shape[1] != num_channels and eeg.shape[2] == num_channels:
        eeg = np.transpose(eeg, (0, 2, 1))

    if eeg.shape[1] != num_channels:
        raise ValueError(
            f"Expected {num_channels} EEG channels, got shape {eeg.shape}"
        )

    labels = np.array(mat["substate"]).squeeze().astype(int)
    subject_ids = np.array(mat["subindex"]).squeeze().astype(int)

    if len(eeg) != len(labels) or len(eeg) != len(subject_ids):
        raise ValueError(
            "EEGsample, substate, and subindex must have the same number of samples"
        )

    return eeg, labels, subject_ids

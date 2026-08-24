import os

import numpy as np
from scipy.io import loadmat

SEED_VIG_FILES = [
    "1_20151124_noon_2.mat",
    "2_20151106_noon.mat",
    "3_20151024_noon.mat",
    "4_20151105_noon.mat",
    "4_20151107_noon.mat",
    "5_20141108_noon.mat",
    "5_20151012_night.mat",
    "6_20151121_noon.mat",
    "7_20151015_night.mat",
    "8_20151022_noon.mat",
    "9_20151017_night.mat",
    "10_20151125_noon.mat",
    "11_20151024_night.mat",
    "12_20150928_noon.mat",
    "13_20150929_noon.mat",
    "14_20151014_night.mat",
    "15_20151126_night.mat",
    "16_20151128_night.mat",
    "17_20150925_noon.mat",
    "18_20150926_noon.mat",
    "19_20151114_noon.mat",
    "20_20151129_night.mat",
    "21_20151016_noon.mat",
]


def create_tri_class_labels(perclos_values):
    """
    Convert PERCLOS values to tri-class labels:
    [0, 0.35] -> 0 (alert)
    (0.35, 0.7] -> 1 (semi-fatigue)
    (0.7, 1] -> 2 (fatigue)
    """
    labels = np.zeros_like(perclos_values)
    labels[(perclos_values > 0.35) & (perclos_values <= 0.7)] = 1
    labels[perclos_values > 0.7] = 2
    return labels.astype(int)


def load_seed_vig(data_dir, num_channels):
    """
    Load SEED-VIG raw EEG and PERCLOS labels.

    Expects:
        data_dir/Raw_Data/*.mat
        data_dir/perclos_labels/*.mat
    """
    raw_dir = os.path.join(data_dir, "Raw_Data")
    perclos_dir = os.path.join(data_dir, "perclos_labels")

    if not os.path.isdir(raw_dir) or not os.path.isdir(perclos_dir):
        raise FileNotFoundError(
            "SEED-VIG data not found. Expected "
            f"{raw_dir} and {perclos_dir}. "
            "See README.md for the official download and directory layout."
        )

    all_eeg = []
    all_y = []
    subject_ids = []

    for idx, filename in enumerate(SEED_VIG_FILES):
        raw_path = os.path.join(raw_dir, filename)
        perclos_path = os.path.join(perclos_dir, filename)

        if not os.path.isfile(raw_path):
            raise FileNotFoundError(f"Missing raw EEG file: {raw_path}")
        if not os.path.isfile(perclos_path):
            raise FileNotFoundError(f"Missing PERCLOS file: {perclos_path}")

        raw = loadmat(raw_path)
        perclos = loadmat(perclos_path)

        raw_eeg = np.array(raw["EEG"]["data"][0][0]).transpose()
        perclos_values = np.array(perclos["perclos"], dtype=float).squeeze()
        n_segments = len(perclos_values)

        eeg_segments = np.array_split(raw_eeg, n_segments, axis=1)
        eeg_data = np.stack(eeg_segments, axis=0)
        y = create_tri_class_labels(perclos_values).squeeze()

        if eeg_data.shape[1] != num_channels:
            raise ValueError(
                f"{filename}: expected {num_channels} EEG channels, "
                f"got {eeg_data.shape[1]}"
            )

        all_eeg.append(eeg_data)
        all_y.append(y)
        subject_ids.extend([idx] * len(y))

    eeg = np.concatenate(all_eeg, axis=0)
    labels = np.concatenate(all_y, axis=0)
    subject_ids = np.array(subject_ids)

    return eeg, labels, subject_ids

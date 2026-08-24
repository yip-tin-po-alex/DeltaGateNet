import numpy as np


def split_indices(subject_ids, mode, fold, n_folds=5, val_ratio=0.2):
    """
    Intra-subject: split each subject's samples into n folds.
    Inter-subject: hold out entire subjects.

    Randomness uses the process-wide numpy RNG (seeded by train.train.set_seed).
    """
    indices = np.arange(len(subject_ids))

    if mode == "intra":
        print(f"Running {n_folds}-Fold Intra-Subject Evaluation - Fold {fold + 1}/{n_folds}")

        train_indices = []
        test_indices = []

        unique_subjects = np.unique(subject_ids)
        for subj in unique_subjects:
            subj_indices = indices[subject_ids == subj]
            np.random.shuffle(subj_indices)

            fold_size = len(subj_indices) // n_folds
            fold_start = fold * fold_size
            fold_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(subj_indices)

            test_indices.extend(subj_indices[fold_start:fold_end])
            train_indices.extend(
                np.concatenate(
                    [
                        subj_indices[:fold_start],
                        subj_indices[fold_end:],
                    ]
                )
            )

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        np.random.shuffle(train_indices)
        val_size = int(val_ratio * len(train_indices))
        val_indices = train_indices[:val_size]
        train_indices = train_indices[val_size:]

    elif mode == "inter":
        print(f"Running {n_folds}-Fold Inter-Subject Evaluation - Fold {fold + 1}/{n_folds}")

        unique_subjects = np.unique(subject_ids)
        np.random.shuffle(unique_subjects)

        fold_size = len(unique_subjects) // n_folds
        fold_start = fold * fold_size
        fold_end = (
            (fold + 1) * fold_size if fold < n_folds - 1 else len(unique_subjects)
        )

        test_subjects = unique_subjects[fold_start:fold_end]
        remaining_subjects = np.concatenate(
            [
                unique_subjects[:fold_start],
                unique_subjects[fold_end:],
            ]
        )

        np.random.shuffle(remaining_subjects)
        split_point = int(0.8 * len(remaining_subjects))
        train_subjects = remaining_subjects[:split_point]
        val_subjects = remaining_subjects[split_point:]

        train_indices = indices[np.isin(subject_ids, train_subjects)]
        val_indices = indices[np.isin(subject_ids, val_subjects)]
        test_indices = indices[np.isin(subject_ids, test_subjects)]

        print(f"Train subjects: {train_subjects}")
        print(f"Val subjects: {val_subjects}")
        print(f"Test subjects: {test_subjects}")
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'intra' or 'inter'.")

    return train_indices, val_indices, test_indices

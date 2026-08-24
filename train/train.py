import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from data import DATASET_DEFAULTS, load_sadt, load_seed_vig, split_indices
from evaluation import evaluate_model, plot_training_history
from model import DeltaGateNet

SEED = 2026
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
BATCH_SIZE = 32


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_dataset(dataset, data_dir, num_channels):
    if dataset == "seed-vig":
        return load_seed_vig(data_dir, num_channels)
    if dataset == "sadt":
        return load_sadt(data_dir, num_channels)
    raise ValueError(f"Unknown dataset '{dataset}'. Use 'seed-vig' or 'sadt'.")


def train_model(model, train_loader, val_loader, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float("inf")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_eeg, batch_y in train_loader:
            batch_eeg = batch_eeg.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_eeg)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_eeg, batch_y in val_loader:
                batch_eeg = batch_eeg.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_eeg)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(batch_y.cpu().numpy().flatten())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        accuracy = accuracy_score(all_labels, all_preds)
        val_accuracies.append(accuracy)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}]: "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Accuracy: {accuracy:.4f}"
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Loaded Best DeltaGateNet Model.")

    return train_losses, val_losses, val_accuracies


def main(dataset, data_dir, num_channels, num_classes, mode="intra", fold=0, output_dir="./logs"):
    set_seed(SEED)

    all_eeg, all_y, subject_ids = load_dataset(dataset, data_dir, num_channels)
    train_indices, val_indices, test_indices = split_indices(subject_ids, mode=mode, fold=fold)

    eeg_train = all_eeg[train_indices]
    y_train = all_y[train_indices]
    eeg_val = all_eeg[val_indices]
    y_val = all_y[val_indices]
    eeg_test = all_eeg[test_indices]
    y_test = all_y[test_indices]

    train_dataset = TensorDataset(
        torch.from_numpy(eeg_train).float(),
        torch.from_numpy(y_train).long(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(eeg_val).float(),
        torch.from_numpy(y_val).long(),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(eeg_test).float(),
        torch.from_numpy(y_test).long(),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(
        f"Train samples: {len(train_dataset)}, "
        f"Val samples: {len(val_dataset)}, "
        f"Test samples: {len(test_dataset)}"
    )
    print(f"Train class distribution: {np.bincount(y_train, minlength=num_classes)}")
    print(f"Val class distribution: {np.bincount(y_val, minlength=num_classes)}")
    print(f"Test class distribution: {np.bincount(y_test, minlength=num_classes)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeltaGateNet(num_channels=num_channels, num_classes=num_classes).to(device)

    fold_dir = os.path.join(output_dir, dataset, mode, f"fold_{fold + 1}")
    checkpoint_path = os.path.join(fold_dir, "best_model.pth")

    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, checkpoint_path
    )

    print(f"\nPlotting training history for Fold {fold + 1}...")
    plot_training_history(
        train_losses,
        val_losses,
        val_accuracies,
        mode=mode,
        fold=fold,
        save_path=os.path.join(fold_dir, "training_history.png"),
    )

    accuracy, precision, recall, f1 = evaluate_model(
        model,
        test_loader,
        num_classes=num_classes,
        save_path=os.path.join(fold_dir, "confusion_matrix.png"),
    )

    return accuracy, precision, recall, f1


def run_cross_validation(
    dataset,
    data_dir,
    num_channels,
    num_classes,
    mode="intra",
    n_folds=5,
    output_dir="./logs",
):
    set_seed(SEED)

    all_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    for fold in range(n_folds):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold + 1}/{n_folds}")
        print("=" * 60)

        accuracy, precision, recall, f1 = main(
            dataset=dataset,
            data_dir=data_dir,
            num_channels=num_channels,
            num_classes=num_classes,
            mode=mode,
            fold=fold,
            output_dir=output_dir,
        )

        all_metrics["accuracy"].append(accuracy)
        all_metrics["precision"].append(precision)
        all_metrics["recall"].append(recall)
        all_metrics["f1"].append(f1)

    print(f"\n{'=' * 60}")
    print(f"{mode.capitalize()}-Subject {n_folds}-Fold Cross Validation Results")
    print("=" * 60)

    for metric_name, values in all_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"Average {metric_name}: {mean_val:.4f} ± {std_val:.4f}")
        print(f"Individual fold {metric_name}s: {[f'{v:.4f}' for v in values]}")

    return all_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeltaGateNet")
    parser.add_argument(
        "--dataset",
        choices=["seed-vig", "sadt"],
        required=True,
        help="Dataset loader to use",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to the dataset directory under datasets/",
    )
    parser.add_argument(
        "--num_channels",
        type=int,
        default=None,
        help="Number of EEG channels (default depends on --dataset)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of classes (default depends on --dataset)",
    )
    parser.add_argument(
        "--mode",
        choices=["intra", "inter"],
        default="intra",
        help="Evaluation protocol",
    )
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output_dir", default="./logs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    defaults = DATASET_DEFAULTS[args.dataset]
    num_channels = args.num_channels if args.num_channels is not None else defaults["num_channels"]
    num_classes = args.num_classes if args.num_classes is not None else defaults["num_classes"]

    run_cross_validation(
        dataset=args.dataset,
        data_dir=args.data_dir,
        num_channels=num_channels,
        num_classes=num_classes,
        mode=args.mode,
        n_folds=args.n_folds,
        output_dir=args.output_dir,
    )

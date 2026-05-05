import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from lru_experiment.common import data_process_util
from lru_experiment.common import validation_util
from lru_experiment.models.lru_native_model import LruNativeModel


os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Expected CSV format: Kaggle oil/gas anomaly detection dataset.
DATA_FILE = PROJECT_ROOT / "data" / "plant_sensor_data.csv"

INPUT_FEATURES = 7
OUTPUT_TARGETS = 1
WINDOW_SIZE = 128
BATCH_SIZE = 128
HIDDEN_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8
POS_WEIGHT = 18.675 # from calculation


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"ROCm compiled? {torch.version.hip is not None}")
    print(f"ROCm version: {torch.version.hip}")

    if device.type == "cuda":
        torch.cuda.init()
    print(f"Device: {device}")

    df = pd.read_csv(DATA_FILE)
    features, labels, _ = data_process_util.prep_lru_data(df)

    split_index = int(len(features) * TRAIN_SPLIT)

    x_train = features[:split_index]
    y_train = labels[:split_index]

    x_val = features[split_index:]
    y_val = labels[split_index:]

    train_loader = data_process_util.create_loader(
        x_train,
        y_train,
        shuffle=True,
        window_size=WINDOW_SIZE,
    )

    val_loader = data_process_util.create_loader(
        x_val,
        y_val,
        shuffle=False,
        window_size=WINDOW_SIZE,
    )

    model = LruNativeModel(
        in_features=INPUT_FEATURES,
        hidden_size=HIDDEN_SIZE,
        out_features=OUTPUT_TARGETS,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHT], device=device),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)

            batch_labels = batch_labels.float()
            batch_labels = batch_labels.view(-1, 1).to(device)

            optimizer.zero_grad()

            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        val_loss, accuracy, precision, recall, f1_score = validation_util.evaluate_model(
            model,
            device,
            val_loader,
            criterion,
        )

    print(
        f"\nEpoch_metrics\n"
        f"train_loss={avg_train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"recall={recall:.4f} | "
        f"f1={f1_score:.4f}"
    )

    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed:.2f}s")

if __name__ == "__main__":
    main()
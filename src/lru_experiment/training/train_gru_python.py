import os
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch import optim

from lru_experiment.common import data_process_util
from lru_experiment.common import validation_util

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Expected CSV format: Kaggle oil/gas anomaly detection dataset.
DATA_FILE = PROJECT_ROOT / "data" / "plant_sensor_data.csv"

HIDDEN_SIZE = 256
WINDOW_SIZE = 300
BATCH_SIZE = 128
FEATURE_SIZE = 7
TRAIN_SPLIT = 0.8
POS_WEIGHT = 18.675   # from calculation
LEARNING_RATE = 0.001
EPOCHS = 20

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        gru_output, h_n = self.gru(x)
        final_hidden = h_n[-1]

        logits = self.out(final_hidden)
        return logits


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #for benchmarking 
    #device = "cpu"
    print("Using device:", device)
    if (device=="cuda"):
          torch.cuda.init()

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

    model = GRUModel(
        input_size=FEATURE_SIZE,
        hidden_size=HIDDEN_SIZE,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHT], device=device),
    )

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            output = model(batch_features)

            loss = criterion(
                output.squeeze(-1),
                batch_labels.float().squeeze(-1),
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(val_loader)

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

    elapsed_time = time.time() - start_time
    print(f"\nTraining time: {elapsed_time:.2f}s")


if __name__ == "__main__":
    main()
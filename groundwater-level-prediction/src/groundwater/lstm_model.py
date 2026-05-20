from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import nn


@dataclass
class LSTMArtifact:
    state_dict: dict
    seq_len: int
    hidden_size: int
    num_layers: int
    scaler_mean: float
    scaler_std: float


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.linear(last)


def _scale(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mean = float(values.mean())
    std = float(values.std()) if float(values.std()) > 1e-8 else 1.0
    scaled = (values - mean) / std
    return scaled, mean, std


def _build_sequences(series: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(series) <= seq_len:
        raise ValueError("Series too short for chosen seq_len")
    xs = []
    ys = []
    for i in range(len(series) - seq_len):
        xs.append(series[i : i + seq_len])
        ys.append(series[i + seq_len])
    return np.array(xs), np.array(ys)


def train_lstm(
    series: np.ndarray,
    seq_len: int = 14,
    hidden_size: int = 32,
    num_layers: int = 1,
    epochs: int = 50,
    lr: float = 1e-3,
    random_state: int = 42,
) -> Tuple[LSTMArtifact, List[float]]:
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.use_deterministic_algorithms(True, warn_only=True)

    series = series.astype(np.float32)
    scaled, mean, std = _scale(series)
    X, y = _build_sequences(scaled, seq_len)
    X_t = torch.tensor(X[:, :, None], dtype=torch.float32)
    y_t = torch.tensor(y[:, None], dtype=torch.float32)

    model = LSTMRegressor(input_size=1, hidden_size=hidden_size, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    losses: List[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().numpy()))

    artifact = LSTMArtifact(
        state_dict=model.state_dict(),
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        scaler_mean=mean,
        scaler_std=std,
    )
    return artifact, losses


def load_lstm(artifact: LSTMArtifact) -> LSTMRegressor:
    model = LSTMRegressor(
        input_size=1,
        hidden_size=artifact.hidden_size,
        num_layers=artifact.num_layers,
    )
    model.load_state_dict(artifact.state_dict)
    model.eval()
    return model


def predict_lstm_steps(
    artifact: LSTMArtifact,
    history: np.ndarray,
    steps: int,
) -> List[float]:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    history = history.astype(np.float32)
    if len(history) < artifact.seq_len:
        raise ValueError(f"Need at least {artifact.seq_len} history values for LSTM forecast")
    scaled = (history - artifact.scaler_mean) / artifact.scaler_std
    seq = list(scaled[-artifact.seq_len :])
    model = load_lstm(artifact)
    preds: List[float] = []
    for _ in range(steps):
        x = torch.tensor(np.array(seq)[None, :, None], dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = float(model(x).cpu().numpy().ravel()[0])
        pred = pred_scaled * artifact.scaler_std + artifact.scaler_mean
        preds.append(pred)
        seq.append(pred_scaled)
        seq = seq[-artifact.seq_len :]
    return preds

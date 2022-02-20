"""using rnns to predict of a sin wave, in forecasting you need to include enough data to capture seasonal trends"""
import os
import sys
from typing import List, Tuple, Collection

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_OBVS: int = 800
TRAIN_SIZE: int = 40


def forecasting_training_batches(sequence, train_size: int) -> List:
    """split a sequence of data into a collection of training sequences with corresponding next point prediction
    label"""
    data: List = []
    for i in range(len(sequence) - train_size):
        train_window = i + train_size
        train = y[i:train_window]
        test = y[train_window:train_window + 1]
        data.append((train, test))
    return data


class LSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 50, out_size: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, out_size)
        self.hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
        pred = self.linear(lstm_out.view(len(seq), -1))
        return pred[-1]


x: torch.Tensor = torch.linspace(0, 799, NUM_OBVS)
y: torch.Tensor = torch.sin(x * 2 * np.pi / 40)

plt.figure(figsize=(13, 4))
plt.plot(x, y)
# plt.show()

train, test = y[:-TRAIN_SIZE], y[-TRAIN_SIZE:]
batches = forecasting_training_batches(sequence=train, train_size=TRAIN_SIZE)
print(batches[0])
print(batches[1])

model = LSTM()
criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

epochs: int = 10
future: int = 40

for i in range(epochs):
    for seq, y_train in batches:
        optimiser.zero_grad()
        model.hidden = (
            torch.zeros(1, 1, model.hidden_size),
            torch.zeros(1, 1, model.hidden_size)
        )
        y_pred = model.forward(seq)
        loss = criterion(y_pred, y_train)

        loss.backward()
        optimiser.step()
    print(f'Epoch: {i}, loss: {loss.item()}')

    train_last_window = train[-TRAIN_SIZE:].tolist()

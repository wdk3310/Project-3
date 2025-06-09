import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

torch.manual_seed(15)

ds = xr.open_dataset(r"C:\Data\gistemp250_GHCNv4.nc")

#zones 
zones = [
    (0, 22.5),
    (22.5, 45),
    (45, 67.5),
    (67.5, 90)
]

#definition of lstm model
def create_lstm_model(input_size=1, hidden_size=64, num_layers=1):
    class LSTMTempModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, 🙂)
    return LSTMTempModel()

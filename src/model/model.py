import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import mlflow
import mlflow.pytorch

# Set device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StockPredictor(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            output_size,
            dropout_prob=0.5
        ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.lstm2 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm1(x, (h0, c0))
        out = self.dropout1(out)
        out = self.fc1(out[:, -1, :])
        out = self.sigmoid(out)
        out, _ = self.lstm2(x, (h0, c0))
        out = self.dropout2(out)
        out = self.fc2(out[:, -1, :])
        return out



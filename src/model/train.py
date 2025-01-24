import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import StockPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_sliding_windows(df, sequence_length, stride):
    num_samples = (len(df) - sequence_length) // stride + 1
    windows = []

    for i in range(0, num_samples * stride, stride):
        window = df.iloc[i:i + sequence_length].values
        windows.append(window)

    return torch.tensor(np.array(windows), dtype=torch.float32)

def train_model(
        model,
        criterion, 
        optimizer,
        train_loader,
        test_loader,
        num_epochs
    ):
    
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment("LSTM Stock Data Regression")
    with mlflow.start_run():
        # Log dos hiperparâmetros do modelo
        mlflow.log_param("intermediate_networks", [
            type(model.lstm1).__name__,
            type(model.dropout1).__name__,
            type(model.fc1).__name__,
            type(model.sigmoid).__name__,
            type(model.lstm2).__name__,
            type(model.dropout2).__name__,
            type(model.fc2).__name__
        ])
        mlflow.log_param("input_size", model.input_size)
        mlflow.log_param("hidden_size", model.hidden_size)
        mlflow.log_param("num_layers", model.num_layers)
        mlflow.log_param("output_size", model.output_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_param("learning_rate", optimizer.param_groups[0]["lr"])

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for i, (sequences, labels) in enumerate(train_loader):
                sequences, labels = sequences.to(device), labels.to(device)

                # Forward pass
                outputs = model(sequences)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
                # Log metrics every end of batch
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                if ((i+1) == len(train_loader)):
                    print("Logging into MLFlow")
                    mlflow.log_metric("train_loss", running_loss / (i+1), step=epoch * len(train_loader) + i)
                

        # Salva o modelo
        mlflow.pytorch.log_model(model, "lstm_stock_data_model")

        # Avalia o modelo
        evaluate_model(model, criterion, test_loader)

def evaluate_model(model, criterion, test_loader):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    average_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {average_test_loss:.4f}")
    mlflow.log_metric("test_loss", average_test_loss)


def main():
    # Constantes
    input_size = 6       # Número de features no dataset      
    hidden_size = 50     # Quantidade de unidades ocultas na LSTM
    num_layers = 2       # Quantidade de camadas LSTM
    output_size = 7      # Previsão dos próximos 7 dias
    num_epochs = 300
    batch_size = 6
    learning_rate = 0.001
    sequence_length = 30  # Tamanho da sequência de entrada
    input_sequence_length = 23  # Tamanho da sequência de treino

    df = pd.read_parquet("data/petr4_10years.parquet/")
    dataset = create_sliding_windows(df, sequence_length, stride=output_size)
    train_size = int(0.85 * dataset.shape[0])
    X_train, y_train = dataset[:train_size, :input_sequence_length, 1:], dataset[:train_size, input_sequence_length:, 0]
    X_test, y_test = dataset[train_size:, :input_sequence_length, 1:], dataset[train_size:, input_sequence_length:, 0]

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = StockPredictor(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_model(
        model,
        criterion,
        optimizer,
        train_loader,
        test_loader,
        num_epochs
    )

    torch.save(model.state_dict(), "src/model/lstm_stock_data_model.pth")
    pass

if __name__ == "__main__":
    main()
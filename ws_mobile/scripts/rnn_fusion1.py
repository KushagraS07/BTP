import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

LOG_DIR = os.path.expanduser('~/ws_mobile/sensor_logs')
WINDOW_SIZE = 10  # paper's T=10

class SmoothingRNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, output_size=2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # take last timestep output
        return out


def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)


def train_rnn():
    ekf_file = os.path.join(LOG_DIR, 'ekf_fused_output.csv')
    df = pd.read_csv(ekf_file)

    data = df[['ekf_x', 'ekf_y']].values.astype(np.float32)

    # Normalize for stable training
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-6
    data_norm = (data - mean) / std

    X, y = create_sequences(data_norm, WINDOW_SIZE)
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    model = SmoothingRNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # using practical lr (paper's 1e-6 is too slow for small dataset)
    criterion = nn.MSELoss()

    epochs = 80
    batch_size = 32
    n_samples = X_tensor.shape[0]

    losses = []

    print(f'Training on {n_samples} sequences for {epochs} epochs...')

    for epoch in range(epochs):
        permutation = torch.randperm(n_samples)
        epoch_loss = 0.0

        for i in range(0, n_samples, batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = X_tensor[indices]
            batch_y = y_tensor[indices]

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / (n_samples // batch_size + 1)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')

    # Generate smoothed predictions for full sequence
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).numpy()

    # Denormalize
    predictions_denorm = predictions * std + mean
    actual_denorm = y * std + mean

    # Save smoothed output
    result_df = pd.DataFrame({
        'rnn_x': predictions_denorm[:, 0],
        'rnn_y': predictions_denorm[:, 1],
        'ekf_x': data[WINDOW_SIZE:, 0],
        'ekf_y': data[WINDOW_SIZE:, 1]
    })
    output_file = os.path.join(LOG_DIR, 'rnn_smoothed_output.csv')
    result_df.to_csv(output_file, index=False)
    print(f'Saved RNN smoothed output to: {output_file}')

    # Plot loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('RNN Training Loss')
    plt.grid(True)
    loss_plot = os.path.join(LOG_DIR, 'rnn_loss_curve.png')
    plt.savefig(loss_plot)
    print(f'Saved loss curve to: {loss_plot}')

    # Plot trajectory comparison
    plt.figure(figsize=(10, 8))
    plt.plot(actual_denorm[:, 0], actual_denorm[:, 1], label='EKF Output', alpha=0.6, color='blue')
    plt.plot(predictions_denorm[:, 0], predictions_denorm[:, 1], label='RNN Smoothed', color='green', linewidth=2)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('RNN Smoothing: EKF vs RNN Output')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    traj_plot = os.path.join(LOG_DIR, 'rnn_trajectory_comparison.png')
    plt.savefig(traj_plot)
    print(f'Saved trajectory plot to: {traj_plot}')

    torch.save(model.state_dict(), os.path.join(LOG_DIR, 'rnn_model.pth'))
    print('Model saved!')


if __name__ == '__main__':
    train_rnn()
    print('RNN training complete!')

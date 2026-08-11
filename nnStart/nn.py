# 
""" Purpose: defines neural network through feedforward layer and uses gradient descent to train. Currently takes random inputs for training
Run with python neuralNetwork/nn.py
- TODO: translate telemetry data into static feature vector in the form of : [Speed, Accel, Slope, MotorTemp, ForceX, ForceY, WindResistance]
"""
# TODO: predict *cumulative* energy consumption instead of per interval

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# ==========================================
# 1. DEFINE THE ANN (FEEDFORWARD MLP)
# ==========================================
class TelemetryStrategyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TelemetryStrategyNet, self).__init__()
        
        # Feedforward layers
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU() # Activation function introduces non-linearity
        
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Data flows strictly forward (Feedforward)
        out = self.layer1(x)
        out = self.relu1(out)
        out = self.layer2(out)
        out = self.relu2(out)
        out = self.output_layer(out)
        return out

# ==========================================
# 2. GENERATE MOCK CSV (IF REAL ONE IS MISSING)
# ==========================================

# finding the exact path between where this Python script is located and where the telemtry_dumps/telemetry_001.csv file is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_filename = os.path.join(script_dir, '..', 'telemetry_dumps', 'telemetry_001.csv')
# create telemetry_dumps directory if it does not exist
os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

if not os.path.exists(csv_filename):
    print(f"'{csv_filename}' not found. Generating a dummy CSV for testing...")
    mock_data = {
        'timestamp_ms': np.arange(1000, 101000, 1000), # 1 sec intervals
        'current_mA': np.random.uniform(500, 5000, 100),
        'voltage_mV': np.random.uniform(11000, 12600, 100),
        'ax_x100': np.random.uniform(-200, 200, 100),
        'ay_x100': np.random.uniform(-200, 200, 100),
        'az_x100': np.random.uniform(-1000, -980, 100), # Gravity mostly
        'amag_x100': np.random.uniform(980, 1020, 100),
        'target_strategy': np.random.uniform(0, 1, 100) # What we want to predict
    }
    pd.DataFrame(mock_data).to_csv(csv_filename, index=False)

# ==========================================
# 3. LOAD, CLEAN, AND PREPROCESS CSV DATA (Incorporating Sandbox Logic)
# TODO: compute input consumption (how much energy is used per second, in Watts/second)
# ==========================================
print("Loading data from CSV...")
df = pd.read_csv(csv_filename)

# a. Forward-fill any missing sensor data (From Sandbox #1)
df.ffill(inplace=True)

# b. Calculate parameters that is not be computed by hardware
    # b.i Power (From Sandbox concept #1)
        # TODO: Determine how hardware data is presented - currently assumes voltage and curent are in mV and mA, so divide by 1,000,000 to get Watts
df['power_watts'] = (df['voltage_mV'] * df['current_mA']) / 1000000.0

    # b.ii Elaspsed time: subtract the very first timestamp from every timestamp
        # (Divide by 1000 to convert milliseconds to seconds)
first_timestamp = df['timestamp_ms'].iloc[0]
df['elapsed_time_s'] = (df['timestamp_ms'] - first_timestamp) / 1000.0

    # b.iii: Delta0time: subtract the previous row's timestamp from the current row
        # df['timestamp_ms'].diff() automatically does (Row_N - Row_N-1)
df['delta_time_s'] = df['timestamp_ms'].diff() / 1000.0
        # The very first row will have a NaN delta-time (since there is no previous row). Fill it with 0.
df['delta_time_s'] = df['delta_time_s'].fillna(0)
    # b.iv Energy consumption (Joules): Power (Watts) * Delta-Time (seconds)
df['energy_consumed_joules'] = df['power_watts'] * df['delta_time_s']

# c. Generate dummy answers if missing
if 'target_strategy' not in df.columns:
    print("\n[WARNING]: 'target_strategy' column not found in CSV.")
    print("Generating a dummy target column so training can proceed...\n")
    df['target_strategy'] = np.random.uniform(0, 1, len(df))

# d. define exact input features
feature_columns = [
    'elapsed_time_s', 'current_mA', 'voltage_mV', 'power_watts', 'energy_consumed_joules',
    'ax_x100', 'ay_x100', 'az_x100', 'amag_x100'
]

# Extract the raw inputs (X) and the target we want to predict (y)
# TODO: ensure actual CSV has a column for the target strategy
X_raw = df[feature_columns].values
y_raw = df['target_strategy'].values

# CRITICAL STEP: Scale the features so they all have a mean of 0 and variance of 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Convert numpy arrays to PyTorch Tensors
X_train = torch.tensor(X_scaled, dtype=torch.float32)
y_train = torch.tensor(y_raw, dtype=torch.float32).view(-1, 1) # Reshape to a column vector

# ==========================================
# 4. SETUP HYPERPARAMETERS
# ==========================================
# assume static feature vector has 8 variables: 'elapsed_time_s', 'current_mA', 'voltage_mV', 'power_watts', 'energy_consumed_joules', 'ax_x100', 'ay_x100', 'az_x100', 'amag_x100' --> = revised columns of data from telemetry_dumps + power + energy consumed
INPUT_FEATURES = 9
# Let's assume 'strategy' is a single continuous value (e.g., Target Throttle %)
OUTPUT_FEATURES = 1 
HIDDEN_NEURONS = 32
LEARNING_RATE = 0.005
EPOCHS = 500

# Initialize the model
model = TelemetryStrategyNet(INPUT_FEATURES, HIDDEN_NEURONS, OUTPUT_FEATURES)

# Define how we calculate error (Mean Squared Error for continuous values)
criterion = nn.MSELoss()

# Define the optimizer (Adam is a highly efficient variant of Gradient Descent)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
print("Starting training...")
for epoch in range(EPOCHS):
    # a. Forward Pass: Pass data through the network
    predictions = model(X_train)
    
    # b. Calculate Loss: How far off were the predictions?
    loss = criterion(predictions, y_train)
    
    # c. Zero Gradients: Clear old gradients from the last step
    optimizer.zero_grad()
    
    # d. Backpropagation: Calculate the gradient of the loss with respect to weights
    loss.backward()
    
    # e. Gradient Descent: Update the weights to minimize the loss
    optimizer.step()
    
    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')

print("Training complete. The network is ready to make predictions.")

# ==========================================
# 6. MAKE A PREDICTION ON NEW DATA
# ==========================================
# Imagine a new line of telemetry just came in via CAN bus/serial
# Order must match: [elapsed_time, current, voltage, power, energy, ax, ay, az, amag]
# TODO: calculate the error with ACTUAL data and compute loss
new_telemetry_raw = np.array([[102.0, 2500, 11800, 29.5, 1.47, 50, -10, -990, 1005]])

# We MUST scale the new data using the EXACT SAME scaler we used for training
new_telemetry_scaled = scaler.transform(new_telemetry_raw)

# Convert to tensor and predict
new_tensor = torch.tensor(new_telemetry_scaled, dtype=torch.float32)
predicted_strategy = model(new_tensor)

print(f"New raw input: {new_telemetry_raw[0]}")
print(f"Predicted Strategy Output: {predicted_strategy.item():.4f}")
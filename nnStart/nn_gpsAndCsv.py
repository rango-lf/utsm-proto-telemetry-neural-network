# 
""" Purpose: defines neural network through feedforward layer and uses gradient descent to train. Currently takes random inputs for training
Run with python neuralNetwork/nn_gpsAndCsv.py
- TODO: translate telemetry data into static feature vector in the form of : [Speed, Accel, Slope, MotorTemp, ForceX, ForceY, WindResistance]
NOTE: this file parses both .csv and .gpx files
-  Uses GPX data for physical features (speed, slope, aero drag)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import xml.etree.ElementTree as ET
import math
from datetime import datetime

# --- Physics Constants ---
rho_air = 1.225
A_aero = 0.8
coeff_aero_drag = 0.2

#validate raw data and ensure that it is correct
def validate_telemetry(df, source_name):
    required_columns = [
        "timestamp_ms",
        "current_mA",
        "voltage_mV",
        "ax_x100",
        "ay_x100",
        "az_x100",
        "amag_x100",
    ]

    if df.empty:
        raise ValueError(f"{source_name}: file contains no telemetry rows")

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{source_name}: missing required columns "
            f"{missing_columns}"
        )

    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(
            f"{source_name}: duplicate columns {duplicates}"
        )

    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    numeric_values = df[required_columns].to_numpy(
        dtype=np.float64
    )

    invalid_rows = ~np.isfinite(numeric_values).all(axis=1)

    if invalid_rows.any():
        csv_rows = (df.index[invalid_rows] + 2).tolist()
        raise ValueError(
            f"{source_name}: missing or invalid values "
            f"at CSV rows {csv_rows[:20]}"
        )

    if df["timestamp_ms"].duplicated().any():
        raise ValueError(
            f"{source_name}: timestamps contain duplicates"
        )

    if not df["timestamp_ms"].is_monotonic_increasing:
        raise ValueError(
            f"{source_name}: timestamps must be increasing"
        )

    if (df["voltage_mV"] <= 0).any():
        raise ValueError(
            f"{source_name}: voltage must be greater than zero"
        )

    return df

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two GPS coordinates."""
    R = 6371000 
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi, d_lambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(d_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

def parse_gpx_to_features(gpx_filename):
    """Parses GPX to DataFrame and calculates physics features."""
    tree = ET.parse(gpx_filename)
    root = tree.getroot()
    ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
    
    data = []
    for trkpt in root.findall('.//gpx:trkpt', ns):
        lat = float(trkpt.attrib['lat'])
        lon = float(trkpt.attrib['lon'])
        ele = float(trkpt.find('gpx:ele', ns).text)
        time_str = trkpt.find('gpx:time', ns).text
        dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
        data.append({'time_dt': dt, 'lat': lat, 'lon': lon, 'ele': ele})
        
    df_gpx = pd.DataFrame(data)
    if df_gpx.empty: return df_gpx
    
    # Normalize time to match CSV elapsed_time_s
    df_gpx['elapsed_time_s'] = (df_gpx['time_dt'] - df_gpx['time_dt'].iloc[0]).dt.total_seconds()
    
    # Calculate Distances & Speed
    df_gpx['dist_step_m'] = 0.0
    for i in range(1, len(df_gpx)):
        df_gpx.loc[i, 'dist_step_m'] = haversine(
            df_gpx.loc[i-1, 'lat'], df_gpx.loc[i-1, 'lon'],
            df_gpx.loc[i, 'lat'], df_gpx.loc[i, 'lon']
        )
        
    df_gpx['dt'] = df_gpx['elapsed_time_s'].diff().fillna(1.0).replace(0, 1.0)
    df_gpx['speed_mps'] = df_gpx['dist_step_m'] / df_gpx['dt']
    
    # Calculate Slope
    df_gpx['ele_diff'] = df_gpx['ele'].diff().fillna(0)
    # math.atan(y/x) - avoiding division by zero
    df_gpx['slope_rad'] = np.arctan(df_gpx['ele_diff'] / df_gpx['dist_step_m'].replace(0, np.nan)).fillna(0)
    
    return df_gpx[['elapsed_time_s', 'speed_mps', 'slope_rad']]


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
# 2. LOCATE TELEMETRY CSV
# ==========================================

script_dir = os.path.dirname(os.path.abspath(__file__))

csv_filename = os.path.join(
    script_dir,
    "..",
    "telemetry_dumps",
    "telemetry_001.csv",
)

if not os.path.exists(csv_filename):
    raise FileNotFoundError(
        f"Telemetry file not found: {csv_filename}"
    )

gpx_filename = os.path.join(
    script_dir, 
    "..", 
    "telemetry_dumps", 
    "telemetry_001.gpx"
)

if not os.path.exists(gpx_filename):
    raise FileNotFoundError(
        f"Telemetry file not found: {gpx_filename}"
    )
# ==========================================
# 3. LOAD, CLEAN, AND PREPROCESS CSV DATA (Incorporating Sandbox Logic)
# TODO: compute input consumption (how much energy is used per second, in Watts/second)
# ==========================================
print("Loading data from CSV...")
# validate data
df = pd.read_csv(csv_filename)
df = validate_telemetry(df, csv_filename)

# b. Calculate parameters that is not be computed by hardware
    # b.i Power (From Sandbox concept #1)
        # TODO: Determine how hardware data is presented - currently assumes voltage and curent are in mV and mA, so divide by 1,000,000 to get Watts
#convert into mV/mA 
df["voltage_v"] = df["voltage_mV"] / 1000.0
df["current_a"] = df["current_mA"] / 1000.0
#calculate power
df["power_watts"] = (
    df["voltage_v"] * df["current_a"]
)
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
# Energy consumed between the previous row and the current row.
average_power_watts = (
    df['power_watts']
    + df['power_watts'].shift(1)
) / 2.0

df['energy_consumed_joules'] = (
    average_power_watts
    * df['delta_time_s']
).fillna(0.0)
df["cumulative_energy_joules"] = (
    df["energy_consumed_joules"].cumsum()
)
# BTW 1 Wh = 3,600 Joules
df["cumulative_energy_wh"] = (
    df["cumulative_energy_joules"] / 3600.0
)

measured_total_joules = (
    df["cumulative_energy_joules"].iloc[-1]
)

measured_total_wh = (
    df["cumulative_energy_wh"].iloc[-1]
)

print(f"Telemetry rows: {len(df)}")
print(
    f"Run duration: "
    f"{df['elapsed_time_s'].iloc[-1]:.3f} seconds"
)
print(
    f"Measured total energy: "
    f"{measured_total_joules:.3f} J"
)
print(
    f"Measured total energy: "
    f"{measured_total_wh:.6f} Wh"
)
# The target for row i is the measured energy in the next interval.
df['target_next_energy_joules'] = (
    df['energy_consumed_joules'].shift(-1)
)

# The final row has no future interval, so it cannot be used for training.
df = df.dropna(
    subset=['target_next_energy_joules']
).reset_index(drop=True)

# d. define exact input features
feature_columns = [
    'elapsed_time_s', 'current_mA', 'voltage_mV', 'power_watts', 'energy_consumed_joules',
    'ax_x100', 'ay_x100', 'az_x100', 'amag_x100'
]

# --- MERGE GPX DATA IF IT EXISTS ---
if os.path.exists(gpx_filename):
    print("GPX file found! Parsing physics features...")
    gpx_df = parse_gpx_to_features(gpx_filename)
    
    # Align high-frequency CSV data with 1Hz GPX data based on elapsed time.
    # Assumes both data recorders were started at roughly the same time.
    df = df.sort_values('elapsed_time_s')
    gpx_df = gpx_df.sort_values('elapsed_time_s')
    
    # merge_asof fills the CSV rows with the most recent previous GPX row
    df = pd.merge_asof(df, gpx_df, on='elapsed_time_s', direction='backward')
    
    # Backfill missing values for any early CSV rows before the first GPS lock
    df['speed_mps'] = df['speed_mps'].bfill().fillna(0)
    df['slope_rad'] = df['slope_rad'].bfill().fillna(0)
    
    # Calculate Aero Drag
    df['aero_drag_N'] = 0.5 * rho_air * A_aero * coeff_aero_drag * (df['speed_mps'] ** 2)
    
    # Add new physics features to the input array
    feature_columns.extend(['speed_mps', 'slope_rad', 'aero_drag_N'])
else:
    print("No GPX file found. Defaulting to raw telemetry.")

df['target_next_energy_joules'] = df['energy_consumed_joules'].shift(-1)
df = df.dropna(subset=['target_next_energy_joules']).reset_index(drop=True)


# Extract the raw inputs (X) and the target we want to predict (y)
# TODO: ensure actual CSV has a column for the target strategy
X_raw = df[feature_columns].values
y_raw = df['target_next_energy_joules'].values

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
INPUT_FEATURES = len(feature_columns) # calculates length based on the # of input names
# The output is predicted next-interval energy in joules.
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
# Prediction array must dynamically match the size of feature_columns depending on if GPX was used.
if 'speed_mps' in feature_columns:
    new_telemetry_raw = np.array([[102.0, 2500, 11800, 29.5, 1.47, 50, -10, -990, 1005, 11.2, 0.05, 12.0]])
else:
    new_telemetry_raw = np.array([[102.0, 2500, 11800, 29.5, 1.47, 50, -10, -990, 1005]])

# We MUST scale the new data using the EXACT SAME scaler we used for training
new_telemetry_scaled = scaler.transform(new_telemetry_raw)

# Convert to tensor and predict
new_tensor = torch.tensor(new_telemetry_scaled, dtype=torch.float32)
model.eval()

with torch.no_grad():
    predicted_next_energy = model(new_tensor)

print(f"New raw input: {new_telemetry_raw[0]}")
print(
    f"Predicted next-interval energy: "
    f"{predicted_next_energy.item():.4f} J"
)

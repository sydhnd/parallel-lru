import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from lru_experiment.common.windowed_sensor_dataset import SensorWindowDataset

def create_loader(features, labels, shuffle, window_size):
    dataset = SensorWindowDataset(
        features,
        labels,
        size=window_size,
    )

    return DataLoader(
        dataset,
        batch_size=window_size,
        shuffle=shuffle,
    )

def prep_lru_data(df):
   
    scaler = MinMaxScaler()    
    features = [
        'temperature', 'pressure', 'flow_rate', 'vibration_level', 
        'valve_position', 'motor_speed', 'chemical_concentration'
    ]
    target = 'anomaly_label'

    x_scaled = scaler.fit_transform(df[features])
    x = torch.tensor(x_scaled, dtype=torch.float32)
    y = torch.tensor(df[target].values, dtype=torch.float32)

    return x, y, scaler

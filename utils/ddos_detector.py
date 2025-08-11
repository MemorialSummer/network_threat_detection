# DDoS 检测模型加载与推理
import torch
import torch.nn as nn
import numpy as np

class DDoSDetector(nn.Module):
    def __init__(self):
        super(DDoSDetector, self).__init__()
        self.fc1 = nn.Linear(8, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def load_ddos_model(model_path):
    model = DDoSDetector()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_ddos(model, features: np.ndarray):
    # features: shape (8,)
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        y_pred = model(x)
        return float(y_pred.item())

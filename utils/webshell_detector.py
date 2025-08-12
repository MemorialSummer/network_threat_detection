# utils/webshell_detector.py
import os
import torch
import torch.nn as nn
import joblib

class MLP(nn.Module):
    def __init__(self, input_dim=200):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 5)
        self.fc2 = nn.Linear(5, 2)
        self.fc3 = nn.Linear(2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def load_webshell_model(model_path):
    model = MLP(input_dim=200)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

def predict_webshell(model, text, vectorizer, svd):
    # 文本转特征
    X_tfidf = vectorizer.transform([text])
    X_reduced = svd.transform(X_tfidf)
    X_tensor = torch.tensor(X_reduced, dtype=torch.float32)
    with torch.no_grad():
        score = model(X_tensor).item()
    return score

def load_webshell_pipeline(model_path, vectorizer_path, svd_path):
    model = load_webshell_model(model_path)
    vectorizer = joblib.load(vectorizer_path)
    svd = joblib.load(svd_path)
    return model, vectorizer, svd

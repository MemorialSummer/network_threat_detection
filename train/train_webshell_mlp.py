# Webshell检测 - MLP+2-gram+TF-IDF
# TODO: 实现Webshell检测的MLP训练脚本
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# 假设你有CSV数据，包含两列：
# 'payload'（webshell代码文本或请求体），'label'（0=正常，1=webshell）

# 1. 读取数据
data = pd.read_csv("webshell_dataset.csv")
texts = data['payload'].astype(str).values
labels = data['label'].values

# 2. 特征工程：2-gram词袋 + TF-IDF
count_vect = CountVectorizer(ngram_range=(2, 2))  # 2-gram词袋
X_counts = count_vect.fit_transform(texts)

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

# 转换为numpy数组
X_features = X_tfidf.toarray()

# 3. 归一化特征（均值0，方差1）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# 4. 转换为tensor
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

# 5. 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42
)

# 6. 定义MLP模型
class WebshellMLP(nn.Module):
    def __init__(self, input_dim):
        super(WebshellMLP, self).__init__()
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

model = WebshellMLP(X_train.shape[1])

# 7. 训练设置
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 20
batch_size = 64

# 8. 训练过程
for epoch in range(epochs):
    permutation = torch.randperm(X_train.size(0))
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 9. 测试准确率
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_label = (y_pred >= 0.5).float()
    acc = (y_pred_label == y_test).sum() / y_test.size(0)
    print(f"Test Accuracy: {acc:.4f}")

# 10. 保存模型及特征处理器
torch.save(model.state_dict(), "../models/webshell_mlp.pth")
import joblib
joblib.dump(count_vect, "../models/webshell_count_vect.pkl")
joblib.dump(tfidf_transformer, "../models/webshell_tfidf_transformer.pkl")
joblib.dump(scaler, "../models/webshell_scaler.pkl")

print("✅ Webshell模型及特征处理器保存完成")
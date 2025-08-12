# Webshell检测 - MLP+2-gram+TF-IDF
# TODO: 实现Webshell检测的MLP训练脚本
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import joblib  # [新增] 保存TF-IDF和SVD

# ========== 1. 数据加载 ==========
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pos_dir = os.path.join(root_dir, "data", "webshell_PHP")
neg_file = os.path.join(root_dir, "data", "No_webshell", "wordpress_php.txt")

texts = []
labels = []

def read_file(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except:
        return ""

# Positive: 所有 A-Z 子文件夹下的 .txt
pos_files = glob.glob(os.path.join(pos_dir, "*", "*.txt"))
for f in pos_files:
    texts.append(read_file(f))
    labels.append(1)

# Negative: 只有一个 txt 文件
neg_text = read_file(neg_file)
# 如果需要多个负样本，可以按行切分
neg_samples = neg_text.split("\n\n")  # 用空行分割成片段
for sample in neg_samples:
    if sample.strip():
        texts.append(sample)
        labels.append(0)

print(f"[INFO] Positive: {len([l for l in labels if l == 1])}, "
      f"Negative: {len([l for l in labels if l == 0])}")

# ========== 2. 特征工程 ==========
vectorizer = TfidfVectorizer(ngram_range=(2, 2), max_features=3000)
X_tfidf = vectorizer.fit_transform(texts)

# 保存TF-IDF向量器
joblib.dump(vectorizer, os.path.join(root_dir, "models", "webshell_vectorizer.pkl"))

# SVD 降维到 200维
svd = TruncatedSVD(n_components=200, random_state=42)
X_reduced = svd.fit_transform(X_tfidf)

# 保存SVD模型
joblib.dump(svd, os.path.join(root_dir, "models", "webshell_svd.pkl"))

X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, labels, test_size=0.2, random_state=42, stratify=labels
)

# 转成 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# ========== 3. 定义 MLP 模型 ==========
class MLP(nn.Module):
    def __init__(self, input_dim):
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

model = MLP(input_dim=200)

# ========== 4. 训练配置 ==========
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ========== 5. 训练循环 ==========
for epoch in range(10):  # 10轮即可
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# ========== 6. 测试集评估 ==========
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch).squeeze()
        preds = (outputs >= 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

print(f"Test Accuracy: {correct/total:.4f}")

# ========== 7. 保存模型 ==========
os.makedirs(os.path.join(root_dir, "models"), exist_ok=True)
model_path = os.path.join(root_dir, "models", "webshell_mlp.pth")
torch.save(model.state_dict(), model_path)
print(f"[INFO] 模型已保存到 {model_path}")
# DDoS检测 - ANN实现
# TODO: 实现DDoS检测的人工神经网络训练脚本
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# =======================
# 1. 读取数据
# DDoS 检测中常用的流量特征（你可以从抓包数据统计得到）：
# src_pkt_rate — 源IP每秒发送的包数（Packets per second from source IP）
# dst_pkt_rate — 目的IP每秒接收的包数（Packets per second to destination IP）
# avg_pkt_size — 平均包大小（字节）
# flow_duration — 流持续时间（秒）
# protocol_type — 协议类型（TCP=0, UDP=1, ICMP=2 等 One-Hot 或编码）
# src_port_entropy — 源端口的熵（反映端口随机性）
# dst_port_entropy — 目的端口的熵（反映扫描/攻击特征）
# tcp_flag_count — TCP 标志位计数（SYN、ACK 等比例）
# =======================
data = pd.read_csv("data/ddos_dataset.csv")

# 从数据集中提取与8维输入对应的特征
# 1. src_pkt_rate   → pktrate
# 2. dst_pkt_rate   → Pairflow（假设表示目的方向的包速率）
# 3. avg_pkt_size   → bytecount / pktcount
# 4. flow_duration  → tot_dur
# 5. protocol_type  → Protocol（需整数编码）
# 6. src_port_entropy → tx_bytes / (tx_bytes+rx_bytes+1) 作为端口使用分布代理
# 7. dst_port_entropy → rx_bytes / (tx_bytes+rx_bytes+1) 作为端口使用分布代理
# 8. tcp_flag_count → flows（假设作为标志数量的代理值）

# 计算平均包大小
data["avg_pkt_size"] = data["bytecount"] / (data["pktcount"] + 1e-6)

# 简单编码协议类型（假设 Protocol 是字符串）
protocol_map = {p: idx for idx, p in enumerate(data["Protocol"].unique())}
data["protocol_type"] = data["Protocol"].map(protocol_map)

# 计算源/目的端口熵的近似值（用 tx_bytes, rx_bytes 比例代替）
data["src_port_entropy"] = data["tx_bytes"] / (data["tx_bytes"] + data["rx_bytes"] + 1e-6)
data["dst_port_entropy"] = data["rx_bytes"] / (data["tx_bytes"] + data["rx_bytes"] + 1e-6)

# 按8维特征顺序选择列
features = [
    "pktrate",             # 1
    "Pairflow",            # 2
    "avg_pkt_size",        # 3
    "tot_dur",             # 4
    "protocol_type",       # 5
    "src_port_entropy",    # 6
    "dst_port_entropy",    # 7
    "flows"                # 8
]
X = data[features].values
y = data["label"].values

# 标准化 & 转换 Tensor（和原来一致）
scaler = StandardScaler()
X = scaler.fit_transform(X)
import joblib
joblib.dump(scaler, "models/ddos_scaler.save")
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
# =======================
# 1. 读取数据
# DDoS 检测中常用的流量特征（你可以从抓包数据统计得到）：
# src_pkt_rate — 源IP每秒发送的包数（Packets per second from source IP）
# dst_pkt_rate — 目的IP每秒接收的包数（Packets per second to destination IP）
# avg_pkt_size — 平均包大小（字节）
# flow_duration — 流持续时间（秒）
# protocol_type — 协议类型（TCP=0, UDP=1, ICMP=2 等 One-Hot 或编码）
# src_port_entropy — 源端口的熵（反映端口随机性）
# dst_port_entropy — 目的端口的熵（反映扫描/攻击特征）
# tcp_flag_count — TCP 标志位计数（SYN、ACK 等比例）
#
#
# src_pkt_rate, dst_pkt_rate, avg_pkt_size, flow_duration,
# protocol_type, src_port_entropy, dst_port_entropy, tcp_flag_count, label
# =======================

# 2.4 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =======================
# 3. 定义 ANN 模型
# 输入层: 8个神经元
# 隐藏层1: 100个神经元
# 隐藏层2: 100个神经元
# 输出层: 1个神经元 (Sigmoid激活，用于二分类)
# =======================
class DDoSDetector(nn.Module):
    def __init__(self):
        super(DDoSDetector, self).__init__()
        self.fc1 = nn.Linear(8, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))   # 隐藏层1 + ReLU激活
        x = self.relu(self.fc2(x))   # 隐藏层2 + ReLU激活
        x = self.sigmoid(self.fc3(x))# 输出层 + Sigmoid激活
        return x

# =======================
# 4. 训练模型
# =======================
model = DDoSDetector()
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
batch_size = 64

for epoch in range(epochs):
    permutation = torch.randperm(X_train.size()[0])
    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        
        # 前向传播
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 每轮打印一次loss
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# =======================
# 5. 测试模型
# =======================
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_cls = (y_pred >= 0.5).float()
    acc = (y_pred_cls == y_test).sum() / y_test.size(0)
    print(f"Test Accuracy: {acc:.4f}")

# =======================
# 6. 保存模型
# =======================
torch.save(model.state_dict(), "models/ddos_ann.pth")
print("✅ 模型已保存到 models/ddos_ann.pth")
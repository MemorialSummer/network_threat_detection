# 网络威胁检测系统

## 目录结构

- main.py: 主检测程序
- capture/: 数据抓包与预处理模块
- train/: 各类威胁检测模型训练脚本
- models/: 训练好的模型文件
- utils/: 工具函数与公共模块

network_threat_detection/
│
├── main.py                         # 主检测程序（抓包、流量解析、调用各模型检测）
│
├── data/                           # 各类模型训练数据
│   └── ...
├── capture/                        # 数据抓包与预处理模块
│   ├── packet_sniffer.py           # 网卡抓包模块
│   ├── feature_extraction.py       # 流量特征提取（不同攻击通用）
│   ├── preprocessing.py            # 数据清洗与格式化
│
├── train/                          # 各类威胁检测模型的训练脚本
│   ├── train_ddos_ann.py           # DDoS检测 - ANN实现
│   ├── train_webshell_mlp.py       # Webshell检测 - MLP+2-gram+TF-IDF
│   ├── train_botnet_lr.py          # 僵尸网络 - 逻辑回归
│   ├── train_botnet_perceptron.py  # 僵尸网络 - 感知机
│   ├── train_botnet_dt.py          # 僵尸网络 - 决策树
│   ├── train_botnet_gnb.py         # 僵尸网络 - 高斯朴素贝叶斯
│   ├── train_botnet_knn.py         # 僵尸网络 - KNN
│   ├── train_sql_injection.py      # SQL注入检测
│   ├── train_xml_injection.py      # XML注入检测
│   ├── train_portscan.py           # 端口扫描检测
│   ├── train_xss.py                # 跨站脚本检测
│   ├── train_malware.py            # 病毒、蠕虫、木马检测
│   ├── train_apt.py                # APT攻击检测
│   ├── train_zeroday.py            # 0day攻击检测
│
├── models/                         # 训练好的模型文件（.pkl / .h5 / .joblib等）
│   ├── ddos_ann.h5
│   ├── webshell_mlp.h5
│   ├── botnet_lr.pkl
│   ├── botnet_perceptron.pkl
│   ├── botnet_dt.pkl
│   ├── botnet_gnb.pkl
│   ├── botnet_knn.pkl
│   ├── sql_injection.pkl
│   ├── xml_injection.pkl
│   ├── portscan.pkl
│   ├── xss.pkl
│   ├── malware.pkl
│   ├── apt.pkl
│   ├── zeroday.pkl
│
├── utils/                          # 工具函数与公共模块
│   ├── data_loader.py              # 数据加载
│   ├── ddos_detector.py            # DDoS流量检测
│   ├── model_loader.py             # 模型统一加载接口
│   ├── metrics.py                  # 模型评估指标
│   ├── logger.py                   # 日志模块
│
├── requirements.txt                # Python依赖
└── README.md                       # 项目说明文档

## 简介
本项目用于网络流量威胁检测，支持多种攻击类型的检测与模型训练。

## 训练数据来源
DDOS：https://www.kaggle.com/datasets/aikenkazin/ddos-sdn-dataset?resource=download

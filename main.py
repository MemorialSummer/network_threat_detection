# 主检测程序
# 网络威胁检测主入口

# 主检测程序
# 网络威胁检测主入口
import joblib
import numpy as np
from utils.ddos_detector import load_ddos_model, predict_ddos
# 其它模型加载接口预留
# from utils.webshell_detector import ...
# from utils.botnet_detector import ...
# ...

MODEL_PATH_DDOS = "models/ddos_ann.pth"

def detect_ddos(features, model, scaler=None):
    if scaler is not None:
        features = scaler.transform([features])[0]
    score = predict_ddos(model, np.array(features))
    if score >= 0.5:
        print(f"[DDoS预警] 检测到疑似DDoS攻击，置信度: {score:.4f}")
    else:
        print(f"[DDoS正常] 置信度: {score:.4f}")

def detect_webshell(features):
    # TODO: 调用Webshell检测模型
    pass

def detect_botnet(features):
    # TODO: 调用僵尸网络检测模型
    pass

def detect_sql_injection(features):
    # TODO: 调用SQL注入检测模型
    pass

def detect_xml_injection(features):
    # TODO: 调用XML注入检测模型
    pass

def detect_portscan(features):
    # TODO: 调用端口扫描检测模型
    pass

def detect_xss(features):
    # TODO: 调用跨站脚本攻击检测模型
    pass

def detect_malware(features):
    # TODO: 调用病毒/蠕虫/木马检测模型
    pass

def detect_apt(features):
    # TODO: 调用APT攻击检测模型
    pass

def detect_zeroday(features):
    # TODO: 调用0day攻击检测模型
    pass

def main():
    print("加载DDoS检测模型...")
    model_ddos = load_ddos_model(MODEL_PATH_DDOS)
    try:
        scaler_ddos = joblib.load("models/ddos_scaler.save")
    except Exception:
        scaler_ddos = None
        print("警告：未加载scaler，将不进行标准化。建议在训练时保存scaler。")

    # 示例数据，按特征顺序：
    features = [
        11335,      # src_pkt_rate
        1,          # dst_pkt_rate
        509228.2,   # avg_pkt_size (示例: bytecount/pktcount)
        10,         # flow_duration
        1,          # protocol_type (UDP=1)
        0.062,      # src_port_entropy (示例)
        0.938,      # dst_port_entropy (示例)
        3           # tcp_flag_count
    ]
    print("测试样本特征:", features)
    detect_ddos(features, model_ddos, scaler_ddos)
    # 其它威胁检测预留
    # detect_webshell(features)
    # detect_botnet(features)
    # detect_sql_injection(features)
    # detect_xml_injection(features)
    # detect_portscan(features)
    # detect_xss(features)
    # detect_malware(features)
    # detect_apt(features)
    # detect_zeroday(features)

if __name__ == "__main__":
    main()

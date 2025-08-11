# 流量特征提取（不同攻击通用）
# TODO: 实现特征提取逻辑
from collections import defaultdict
import numpy as np
import time

# 这是一个示例实现，需要配合具体抓包代码使用
# 你可以传入一个时间窗口内的流量包列表，每个包是字典形式
# 包含: src_ip, dst_ip, protocol, src_port, dst_port, pkt_len, tcp_flags 等字段

class FeatureExtractor:
    def __init__(self, time_window=1.0):
        self.time_window = time_window  # 秒级时间窗口
        self.reset()
        
    def reset(self):
        # 重置窗口数据
        self.pkts = []
        self.start_time = time.time()
        
    def add_packet(self, pkt):
        # pkt是一个字典，包含包相关信息
        # 例如： {'src_ip':..., 'dst_ip':..., 'protocol':..., 'src_port':..., 'dst_port':..., 'length':..., 'tcp_flags':...}
        self.pkts.append(pkt)
        
    def extract_features(self):
        now = time.time()
        # 过滤时间窗口内的数据包
        window_pkts = [p for p in self.pkts if now - p['timestamp'] <= self.time_window]
        
        if len(window_pkts) == 0:
            # 没数据时返回全0特征
            return np.zeros(8, dtype=np.float32)
        
        # 1. src_pkt_rate (源IP每秒包数)
        src_counts = defaultdict(int)
        for p in window_pkts:
            src_counts[p['src_ip']] += 1
        avg_src_pkt_rate = np.mean(list(src_counts.values())) / self.time_window
        
        # 2. dst_pkt_rate (目的IP每秒包数)
        dst_counts = defaultdict(int)
        for p in window_pkts:
            dst_counts[p['dst_ip']] += 1
        avg_dst_pkt_rate = np.mean(list(dst_counts.values())) / self.time_window
        
        # 3. avg_pkt_size (平均包大小)
        sizes = [p['length'] for p in window_pkts]
        avg_pkt_size = np.mean(sizes)
        
        # 4. flow_duration (时间窗口时长)
        flow_duration = self.time_window
        
        # 5. protocol_type (协议类型编码)
        proto_map = {'TCP':0, 'UDP':1, 'ICMP':2}
        protos = [proto_map.get(p['protocol'], 3) for p in window_pkts]
        avg_proto = np.mean(protos)  # 这里简单取均值作为编码近似
        
        # 6. src_port_entropy (源端口比例)
        src_ports = [p['src_port'] for p in window_pkts]
        src_port_counts = defaultdict(int)
        for sp in src_ports:
            src_port_counts[sp] += 1
        total_src_ports = sum(src_port_counts.values())
        src_port_entropy = 0.0
        for c in src_port_counts.values():
            p = c / total_src_ports
            src_port_entropy -= p * np.log2(p)
        
        # 7. dst_port_entropy (目的端口比例)
        dst_ports = [p['dst_port'] for p in window_pkts]
        dst_port_counts = defaultdict(int)
        for dp in dst_ports:
            dst_port_counts[dp] += 1
        total_dst_ports = sum(dst_port_counts.values())
        dst_port_entropy = 0.0
        for c in dst_port_counts.values():
            p = c / total_dst_ports
            dst_port_entropy -= p * np.log2(p)
        
        # 8. tcp_flag_count (TCP标志位计数，这里用SYN+ACK等数量)
        tcp_flags = [p.get('tcp_flags', '') for p in window_pkts]
        tcp_flag_count = sum(['S' in flags or 'A' in flags for flags in tcp_flags])
        
        features = np.array([
            avg_src_pkt_rate,
            avg_dst_pkt_rate,
            avg_pkt_size,
            flow_duration,
            avg_proto,
            src_port_entropy,
            dst_port_entropy,
            tcp_flag_count
        ], dtype=np.float32)
        
        return features
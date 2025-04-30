import time
from typing import Dict, List
import psutil
import numpy as np

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'cpu': [],
            'memory': [],
            'network': [],
            'disk': []
        }
        self.start_time = time.time()

    def update_metrics(self):
        self.metrics['cpu'].append(psutil.cpu_percent())
        self.metrics['memory'].append(psutil.virtual_memory().percent)
        net_io = psutil.net_io_counters()
        self.metrics['network'].append((net_io.bytes_sent, net_io.bytes_recv))
        self.metrics['disk'].append(psutil.disk_usage('/').percent)

    def get_summary(self) -> Dict[str, float]:
        return {
            'uptime': time.time() - self.start_time,
            'avg_cpu': np.mean(self.metrics['cpu']),
            'max_memory': max(self.metrics['memory']),
            'network_usage': self.metrics['network'][-1]
        }
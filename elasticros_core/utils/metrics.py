#!/usr/bin/env python3
"""
Metrics collection utilities for system monitoring
"""

import psutil
import time
import threading
from collections import deque
from typing import Dict, Optional
import logging

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    
logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects system metrics like CPU, memory, power consumption"""
    
    def __init__(self, history_size: int = 100):
        """
        Initialize metrics collector.
        
        Args:
            history_size: Number of historical data points to keep
        """
        self.history_size = history_size
        
        # Metrics history
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.power_history = deque(maxlen=history_size)
        
        # Monitoring thread
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 1.0  # seconds
        
        # Start monitoring
        self.start()
        
    def start(self):
        """Start background monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Started metrics monitoring")
            
    def stop(self):
        """Stop background monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Stopped metrics monitoring")
            
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # Collect metrics
                cpu = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory().percent
                power = self._estimate_power()
                
                # Store in history
                timestamp = time.time()
                self.cpu_history.append((timestamp, cpu))
                self.memory_history.append((timestamp, memory))
                self.power_history.append((timestamp, power))
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                
            time.sleep(self.monitor_interval)
            
    def _estimate_power(self) -> float:
        """
        Estimate power consumption in watts.
        This is a simplified estimation - real implementation would use
        hardware-specific power monitoring.
        """
        # Base power consumption
        base_power = 10.0  # watts
        
        # CPU contribution (simplified linear model)
        cpu_percent = psutil.cpu_percent(interval=0)
        cpu_power = cpu_percent * 0.5  # 0.5W per percent
        
        # GPU contribution if available
        gpu_power = 0.0
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    # Average GPU power draw
                    gpu_power = sum(gpu.powerDraw for gpu in gpus) / len(gpus)
            except:
                pass
                
        total_power = base_power + cpu_power + gpu_power
        return total_power
        
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        return psutil.cpu_percent(interval=0.1)
        
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        return psutil.virtual_memory().percent
        
    def get_power_consumption(self) -> float:
        """Get current estimated power consumption in watts"""
        return self._estimate_power()
        
    def get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage if available"""
        if not GPU_AVAILABLE:
            return None
            
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                # Average GPU utilization
                return sum(gpu.load * 100 for gpu in gpus) / len(gpus)
        except:
            return None
            
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        stats = {
            'cpu_percent': self.get_cpu_usage(),
            'memory_percent': self.get_memory_usage(),
            'power_watts': self.get_power_consumption(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1e9,
        }
        
        # Add GPU stats if available
        gpu_usage = self.get_gpu_usage()
        if gpu_usage is not None:
            stats['gpu_percent'] = gpu_usage
            
        # Add averages from history
        if self.cpu_history:
            recent_cpu = [cpu for _, cpu in list(self.cpu_history)[-10:]]
            stats['cpu_avg_10s'] = sum(recent_cpu) / len(recent_cpu)
            
        return stats
        
    def get_history(self, metric: str = 'cpu') -> list:
        """
        Get historical data for a metric.
        
        Args:
            metric: One of 'cpu', 'memory', 'power'
            
        Returns:
            List of (timestamp, value) tuples
        """
        if metric == 'cpu':
            return list(self.cpu_history)
        elif metric == 'memory':
            return list(self.memory_history)
        elif metric == 'power':
            return list(self.power_history)
        else:
            raise ValueError(f"Unknown metric: {metric}")


def monitor_main():
    """Main function for standalone monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ElasticROS metrics monitor')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Monitoring interval in seconds')
    parser.add_argument('--duration', type=int, default=60,
                       help='Monitoring duration in seconds')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(message)s')
    
    # Create collector
    collector = MetricsCollector()
    collector.monitor_interval = args.interval
    
    # Monitor for specified duration
    logger.info(f"Monitoring system metrics for {args.duration} seconds...")
    start_time = time.time()
    
    try:
        while time.time() - start_time < args.duration:
            stats = collector.get_system_stats()
            logger.info(f"CPU: {stats['cpu_percent']:.1f}%, "
                       f"Memory: {stats['memory_percent']:.1f}%, "
                       f"Power: {stats['power_watts']:.1f}W")
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted")
        
    finally:
        collector.stop()
        
        
if __name__ == '__main__':
    monitor_main()
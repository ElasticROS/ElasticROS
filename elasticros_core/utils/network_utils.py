#!/usr/bin/env python3
"""
Network monitoring and management utilities
"""

import subprocess
import time
import socket
import threading
from typing import Dict, Optional, Tuple
import logging
import requests

try:
    import speedtest
    SPEEDTEST_AVAILABLE = True
except ImportError:
    SPEEDTEST_AVAILABLE = False

logger = logging.getLogger(__name__)


class NetworkMonitor:
    """Monitors network conditions - bandwidth, latency, etc"""
    
    def __init__(self, target_host: str = "8.8.8.8"):
        """
        Initialize network monitor.
        
        Args:
            target_host: Host to ping for latency measurement
        """
        self.target_host = target_host
        
        # Cached measurements
        self.last_bandwidth_test = 0
        self.bandwidth_cache = None
        self.bandwidth_cache_duration = 300  # 5 minutes
        
        # Monitoring thread
        self.monitoring = False
        self.monitor_thread = None
        
        # Current network stats
        self.current_latency = 0.0
        self.current_bandwidth = 0.0
        
        # Start monitoring
        self.start()
        
    def start(self):
        """Start background monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Started network monitoring")
            
    def stop(self):
        """Stop background monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Stopped network monitoring")
            
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # Update latency frequently
                self.current_latency = self._measure_latency()
                
                # Update bandwidth less frequently
                if (time.time() - self.last_bandwidth_test > self.bandwidth_cache_duration):
                    self.current_bandwidth = self._measure_bandwidth()
                    
            except Exception as e:
                logger.error(f"Error in network monitoring: {e}")
                
            time.sleep(5)  # Check every 5 seconds
            
    def _measure_latency(self) -> float:
        """Measure network latency in milliseconds"""
        try:
            # Use ping command
            if self._is_linux():
                cmd = ['ping', '-c', '1', '-W', '1', self.target_host]
            else:
                cmd = ['ping', '-n', '1', '-w', '1000', self.target_host]
                
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse ping output
                output = result.stdout
                
                # Linux format: time=X.X ms
                if 'time=' in output:
                    time_str = output.split('time=')[1].split()[0]
                    return float(time_str)
                    
                # Windows format: time<Xms or time=Xms
                elif 'time' in output and 'ms' in output:
                    import re
                    match = re.search(r'time[<=](\d+)ms', output)
                    if match:
                        return float(match.group(1))
                        
        except Exception as e:
            logger.debug(f"Ping failed: {e}")
            
        # Fallback - try socket connection
        try:
            start = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            result = sock.connect_ex((self.target_host, 80))
            sock.close()
            
            if result == 0:
                return (time.time() - start) * 1000
                
        except:
            pass
            
        return 999.0  # High latency on failure
        
    def _measure_bandwidth(self) -> float:
        """Measure network bandwidth in Mbps"""
        # Try speedtest-cli first
        if SPEEDTEST_AVAILABLE:
            try:
                st = speedtest.Speedtest()
                st.get_best_server()
                
                # Download speed in Mbps
                download_speed = st.download() / 1e6
                
                self.last_bandwidth_test = time.time()
                self.bandwidth_cache = download_speed
                
                logger.info(f"Measured bandwidth: {download_speed:.1f} Mbps")
                return download_speed
                
            except Exception as e:
                logger.warning(f"Speedtest failed: {e}")
                
        # Fallback - estimate from file download
        try:
            # Download a small file from a fast CDN
            url = "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png"
            
            start_time = time.time()
            response = requests.get(url, timeout=5)
            end_time = time.time()
            
            if response.status_code == 200:
                # Calculate speed
                file_size = len(response.content)  # bytes
                duration = end_time - start_time
                
                # Convert to Mbps
                speed_mbps = (file_size * 8) / (duration * 1e6)
                
                # This is a rough estimate, scale up
                estimated_bandwidth = speed_mbps * 10  # rough scaling factor
                
                self.last_bandwidth_test = time.time()
                self.bandwidth_cache = estimated_bandwidth
                
                return estimated_bandwidth
                
        except Exception as e:
            logger.warning(f"Bandwidth estimation failed: {e}")
            
        # Return cached value or default
        if self.bandwidth_cache is not None:
            return self.bandwidth_cache
        else:
            return 10.0  # Default 10 Mbps
            
    def get_latency(self) -> float:
        """Get current network latency in ms"""
        if self.monitoring:
            return self.current_latency
        else:
            return self._measure_latency()
            
    def get_current_bandwidth(self) -> float:
        """Get current bandwidth estimate in Mbps"""
        if self.monitoring:
            return self.current_bandwidth
        else:
            # Use cache if recent
            if (self.bandwidth_cache is not None and 
                time.time() - self.last_bandwidth_test < self.bandwidth_cache_duration):
                return self.bandwidth_cache
            else:
                return self._measure_bandwidth()
                
    def estimate_transfer_time(self, data_size_bytes: int) -> float:
        """
        Estimate time to transfer data.
        
        Args:
            data_size_bytes: Size of data in bytes
            
        Returns:
            Estimated transfer time in seconds
        """
        bandwidth_mbps = self.get_current_bandwidth()
        latency_ms = self.get_latency()
        
        # Convert to consistent units
        bandwidth_bytes_per_sec = (bandwidth_mbps * 1e6) / 8
        
        # Transfer time = latency + data_size / bandwidth
        transfer_time = (latency_ms / 1000) + (data_size_bytes / bandwidth_bytes_per_sec)
        
        return transfer_time
        
    def get_network_stats(self) -> Dict:
        """Get comprehensive network statistics"""
        return {
            'latency_ms': self.get_latency(),
            'bandwidth_mbps': self.get_current_bandwidth(),
            'target_host': self.target_host,
            'last_bandwidth_test': self.last_bandwidth_test,
        }
        
    def _is_linux(self) -> bool:
        """Check if running on Linux"""
        import platform
        return platform.system().lower() == 'linux'


class BandwidthController:
    """
    Controls network bandwidth using wondershaper or similar tools.
    Requires sudo privileges.
    """
    
    def __init__(self, interface: str = None):
        """
        Initialize bandwidth controller.
        
        Args:
            interface: Network interface (e.g., 'eth0', 'wlan0')
        """
        self.interface = interface or self._get_default_interface()
        self.original_limits = None
        
    def _get_default_interface(self) -> str:
        """Get default network interface"""
        try:
            # Try to find default route interface
            result = subprocess.run(
                ['ip', 'route', 'show', 'default'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Parse output: "default via X.X.X.X dev eth0"
                parts = result.stdout.split()
                if 'dev' in parts:
                    idx = parts.index('dev')
                    return parts[idx + 1]
                    
        except:
            pass
            
        # Fallback
        return 'eth0'
        
    def set_bandwidth_limit(self, download_kbps: int, upload_kbps: int):
        """
        Set bandwidth limits.
        
        Args:
            download_kbps: Download limit in kbps
            upload_kbps: Upload limit in kbps
        """
        try:
            # Use wondershaper
            cmd = [
                'sudo', 'wondershaper',
                self.interface,
                str(download_kbps),
                str(upload_kbps)
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            
            if result.returncode == 0:
                logger.info(f"Set bandwidth limits: {download_kbps} kbps down, "
                           f"{upload_kbps} kbps up on {self.interface}")
            else:
                logger.error(f"Failed to set bandwidth limits: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Error setting bandwidth limits: {e}")
            
    def clear_limits(self):
        """Clear all bandwidth limits"""
        try:
            cmd = ['sudo', 'wondershaper', 'clear', self.interface]
            subprocess.run(cmd, capture_output=True)
            logger.info(f"Cleared bandwidth limits on {self.interface}")
            
        except Exception as e:
            logger.error(f"Error clearing bandwidth limits: {e}")
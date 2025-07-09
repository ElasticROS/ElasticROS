#!/usr/bin/env python3
"""
Data transfer utilities for efficient robot-cloud communication
"""

import pickle
import zlib
import base64
import numpy as np
import json
from typing import Any, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataTransfer:
    """Handles data serialization, compression, and transfer optimization"""
    
    def __init__(self, compression_level: int = 6):
        """
        Initialize data transfer utilities.
        
        Args:
            compression_level: zlib compression level (0-9)
        """
        self.compression_level = compression_level
        self.stats = {
            'bytes_sent': 0,
            'bytes_received': 0,
            'compression_ratio_sum': 0,
            'transfer_count': 0
        }
        
    def prepare_for_cloud(self, data: Any) -> Dict[str, Any]:
        """
        Prepare data for transfer to cloud.
        Handles serialization and compression.
        
        Args:
            data: Data to send
            
        Returns:
            Dictionary with serialized data and metadata
        """
        original_size = 0
        data_type = type(data).__name__
        
        # Handle different data types
        if isinstance(data, np.ndarray):
            # Numpy arrays - efficient serialization
            original_size = data.nbytes
            
            # Downsample large arrays if needed
            if data.size > 1e6:  # More than 1M elements
                data = self._downsample_array(data)
                
            # Serialize
            serialized = {
                'array': data.tobytes(),
                'dtype': str(data.dtype),
                'shape': data.shape
            }
            serialized = pickle.dumps(serialized)
            
        elif isinstance(data, (dict, list)):
            # JSON-serializable data
            serialized = json.dumps(data).encode('utf-8')
            original_size = len(serialized)
            
        else:
            # General Python objects
            serialized = pickle.dumps(data)
            original_size = len(serialized)
            
        # Compress
        compressed = zlib.compress(serialized, level=self.compression_level)
        compressed_size = len(compressed)
        
        # Encode for safe transfer
        encoded = base64.b64encode(compressed).decode('utf-8')
        
        # Update stats
        self.stats['bytes_sent'] += compressed_size
        self.stats['compression_ratio_sum'] += original_size / compressed_size
        self.stats['transfer_count'] += 1
        
        logger.debug(f"Prepared {data_type} for cloud: "
                    f"{original_size} -> {compressed_size} bytes "
                    f"({compressed_size/original_size:.1%} of original)")
        
        return {
            'data': encoded,
            'data_type': data_type,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'is_downsampled': original_size > 1e6
        }
        
    def receive_from_cloud(self, transfer_dict: Dict[str, Any]) -> Any:
        """
        Receive and reconstruct data from cloud.
        
        Args:
            transfer_dict: Dictionary from prepare_for_cloud
            
        Returns:
            Reconstructed data
        """
        try:
            # Decode
            compressed = base64.b64decode(transfer_dict['data'])
            
            # Decompress
            serialized = zlib.decompress(compressed)
            
            # Update stats
            self.stats['bytes_received'] += len(compressed)
            
            # Deserialize based on type
            data_type = transfer_dict.get('data_type', 'unknown')
            
            if data_type == 'ndarray':
                # Numpy array reconstruction
                array_dict = pickle.loads(serialized)
                data = np.frombuffer(
                    array_dict['array'],
                    dtype=array_dict['dtype']
                ).reshape(array_dict['shape'])
                
            elif data_type in ['dict', 'list']:
                # JSON data
                data = json.loads(serialized.decode('utf-8'))
                
            else:
                # General Python object
                data = pickle.loads(serialized)
                
            logger.debug(f"Received {data_type} from cloud: "
                        f"{len(compressed)} bytes")
            
            return data
            
        except Exception as e:
            logger.error(f"Error receiving data from cloud: {e}")
            raise
            
    def _downsample_array(self, array: np.ndarray) -> np.ndarray:
        """
        Downsample large arrays for faster transfer.
        
        Args:
            array: Input array
            
        Returns:
            Downsampled array
        """
        # Simple downsampling - can be made more sophisticated
        if len(array.shape) == 2:
            # 2D array (e.g., grayscale image)
            return array[::2, ::2]
            
        elif len(array.shape) == 3:
            # 3D array (e.g., color image)
            return array[::2, ::2, :]
            
        else:
            # 1D or higher dimensional
            return array[::2]
            
    def optimize_for_bandwidth(self, 
                             data: Any,
                             available_bandwidth_mbps: float) -> Dict[str, Any]:
        """
        Optimize data for available bandwidth.
        
        Args:
            data: Data to transfer
            available_bandwidth_mbps: Current bandwidth in Mbps
            
        Returns:
            Optimized transfer dictionary
        """
        # Adjust compression based on bandwidth
        if available_bandwidth_mbps < 1:
            # Very low bandwidth - maximum compression
            self.compression_level = 9
        elif available_bandwidth_mbps < 10:
            # Low bandwidth - high compression
            self.compression_level = 7
        else:
            # Good bandwidth - balanced compression
            self.compression_level = 5
            
        # Prepare data with adjusted compression
        transfer_dict = self.prepare_for_cloud(data)
        
        # Add bandwidth hint
        transfer_dict['bandwidth_mbps'] = available_bandwidth_mbps
        
        return transfer_dict
        
    def get_stats(self) -> Dict[str, Any]:
        """Get transfer statistics"""
        stats = self.stats.copy()
        
        # Calculate average compression ratio
        if stats['transfer_count'] > 0:
            stats['avg_compression_ratio'] = (
                stats['compression_ratio_sum'] / stats['transfer_count']
            )
        else:
            stats['avg_compression_ratio'] = 1.0
            
        # Convert bytes to MB
        stats['mb_sent'] = stats['bytes_sent'] / 1e6
        stats['mb_received'] = stats['bytes_received'] / 1e6
        
        return stats
        
    def reset_stats(self):
        """Reset transfer statistics"""
        self.stats = {
            'bytes_sent': 0,
            'bytes_received': 0,
            'compression_ratio_sum': 0,
            'transfer_count': 0
        }


class StreamingDataTransfer(DataTransfer):
    """
    Extended transfer utilities for streaming data (e.g., video).
    Implements chunking and progressive transfer.
    """
    
    def __init__(self, chunk_size: int = 65536, **kwargs):
        """
        Initialize streaming transfer.
        
        Args:
            chunk_size: Size of each chunk in bytes
            **kwargs: Arguments for parent class
        """
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        
    def prepare_stream(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Prepare data for streaming transfer.
        
        Args:
            data: Large data array
            
        Returns:
            Stream metadata
        """
        # Calculate chunks
        data_bytes = data.tobytes()
        total_size = len(data_bytes)
        num_chunks = (total_size + self.chunk_size - 1) // self.chunk_size
        
        # Create stream ID
        import uuid
        stream_id = str(uuid.uuid4())
        
        # Store stream info
        stream_info = {
            'stream_id': stream_id,
            'total_size': total_size,
            'num_chunks': num_chunks,
            'chunk_size': self.chunk_size,
            'dtype': str(data.dtype),
            'shape': data.shape
        }
        
        # Store data for chunking (in practice, might use memory mapping)
        self._active_streams = getattr(self, '_active_streams', {})
        self._active_streams[stream_id] = data_bytes
        
        return stream_info
        
    def get_chunk(self, stream_id: str, chunk_index: int) -> Optional[bytes]:
        """
        Get a specific chunk from stream.
        
        Args:
            stream_id: Stream identifier
            chunk_index: Index of chunk to retrieve
            
        Returns:
            Chunk data or None if not found
        """
        if not hasattr(self, '_active_streams'):
            return None
            
        if stream_id not in self._active_streams:
            return None
            
        data_bytes = self._active_streams[stream_id]
        
        start = chunk_index * self.chunk_size
        end = min(start + self.chunk_size, len(data_bytes))
        
        if start >= len(data_bytes):
            return None
            
        return data_bytes[start:end]
        
    def close_stream(self, stream_id: str):
        """Clean up stream data"""
        if hasattr(self, '_active_streams') and stream_id in self._active_streams:
            del self._active_streams[stream_id]
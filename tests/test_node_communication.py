#!/usr/bin/env python3
"""
Tests for node communication in ElasticROS
"""

import pytest
import numpy as np
import time
import threading
import queue
from unittest.mock import Mock, patch, MagicMock
import json

from elasticros_core import ElasticNode, PressNode, ReleaseNode
from elasticros_core.utils import DataTransfer


class TestNodeCommunication:
    """Test cases for Press-Release node communication"""
    
    @pytest.fixture
    def data_transfer(self):
        """Create DataTransfer instance"""
        return DataTransfer(compression_level=6)
        
    @pytest.fixture
    def mock_press_node(self):
        """Create mock Press node"""
        class MockPressNode(PressNode):
            def _initialize(self):
                self.processed_data = []
                
            def _process(self, data, compute_ratio):
                result = {
                    'data': f"Processed({compute_ratio:.1f}): {data}",
                    'compute_ratio': compute_ratio,
                    'node': 'press'
                }
                self.processed_data.append(result)
                return result
                
        return MockPressNode("mock_press")
        
    @pytest.fixture
    def mock_release_node(self):
        """Create mock Release node"""
        class MockReleaseNode(ReleaseNode):
            def _initialize(self):
                self.processed_data = []
                self.latency = 0.1  # Simulate network latency
                
            def _process(self, data, compute_ratio):
                time.sleep(self.latency)  # Simulate processing time
                result = {
                    'data': f"Cloud processed: {data}",
                    'compute_ratio': compute_ratio,
                    'node': 'release'
                }
                self.processed_data.append(result)
                return result
                
        return MockReleaseNode("mock_release")
        
    def test_basic_communication(self, mock_press_node, mock_release_node):
        """Test basic communication between nodes"""
        # Create elastic node
        elastic = ElasticNode()
        elastic.register_node_pair("test", mock_press_node, mock_release_node)
        
        # Execute task
        result = elastic.elastic_execute("test", "Hello ElasticROS")
        
        # Verify result
        assert result is not None
        assert 'node' in result
        assert result['node'] in ['press', 'release']
        
    def test_data_serialization(self, data_transfer):
        """Test data serialization and deserialization"""
        # Test different data types
        test_cases = [
            # Numpy arrays
            np.random.randn(100, 100),
            np.array([1, 2, 3, 4, 5]),
            
            # Python objects
            {'key': 'value', 'number': 42, 'list': [1, 2, 3]},
            ['item1', 'item2', 'item3'],
            'simple string',
            42.0
        ]
        
        for original_data in test_cases:
            # Serialize
            packed = data_transfer.prepare_for_cloud(original_data)
            
            # Verify packed format
            assert 'data' in packed
            assert 'data_type' in packed
            assert 'original_size' in packed
            assert 'compressed_size' in packed
            
            # Verify compression
            assert packed['compressed_size'] <= packed['original_size']
            
            # Deserialize
            unpacked = data_transfer.receive_from_cloud(packed)
            
            # Verify data integrity
            if isinstance(original_data, np.ndarray):
                np.testing.assert_array_almost_equal(original_data, unpacked)
            else:
                assert original_data == unpacked
                
    def test_large_data_transfer(self, data_transfer):
        """Test transfer of large data"""
        # Create large image
        large_image = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
        
        # Measure serialization time
        start = time.time()
        packed = data_transfer.prepare_for_cloud(large_image)
        pack_time = time.time() - start
        
        # Check compression ratio
        compression_ratio = packed['original_size'] / packed['compressed_size']
        assert compression_ratio > 1.0  # Should achieve some compression
        
        # Measure deserialization time
        start = time.time()
        unpacked = data_transfer.receive_from_cloud(packed)
        unpack_time = time.time() - start
        
        # Verify data
        np.testing.assert_array_equal(large_image, unpacked)
        
        # Performance assertions
        assert pack_time < 1.0  # Should pack in under 1 second
        assert unpack_time < 1.0  # Should unpack in under 1 second
        
    def test_concurrent_node_execution(self, mock_press_node, mock_release_node):
        """Test concurrent execution of multiple nodes"""
        elastic = ElasticNode()
        
        # Register multiple node pairs
        for i in range(3):
            press = type(mock_press_node)(f"press_{i}")
            release = type(mock_release_node)(f"release_{i}")
            elastic.register_node_pair(f"task_{i}", press, release)
            
        # Execute tasks concurrently
        results = queue.Queue()
        
        def execute_task(task_name, data):
            result = elastic.elastic_execute(task_name, data)
            results.put((task_name, result))
            
        # Start threads
        threads = []
        for i in range(3):
            t = threading.Thread(
                target=execute_task,
                args=(f"task_{i}", f"data_{i}")
            )
            threads.append(t)
            t.start()
            
        # Wait for completion
        for t in threads:
            t.join()
            
        # Verify results
        assert results.qsize() == 3
        
        while not results.empty():
            task_name, result = results.get()
            assert result is not None
            assert 'node' in result
            
    def test_node_failure_handling(self):
        """Test handling of node failures"""
        # Create failing nodes
        class FailingPressNode(PressNode):
            def _initialize(self):
                pass
                
            def _process(self, data, compute_ratio):
                raise RuntimeError("Press node failed")
                
        class FailingReleaseNode(ReleaseNode):
            def _initialize(self):
                pass
                
            def _process(self, data, compute_ratio):
                raise RuntimeError("Release node failed")
                
        elastic = ElasticNode()
        
        # Test press node failure
        elastic.register_node_pair(
            "fail_press",
            FailingPressNode("fail_press"),
            Mock(spec=ReleaseNode)
        )
        
        with pytest.raises(RuntimeError):
            elastic.elastic_execute("fail_press", "test")
            
        # Test release node failure
        elastic.register_node_pair(
            "fail_release",
            Mock(spec=PressNode),
            FailingReleaseNode("fail_release")
        )
        
        # This might not fail if elastic chooses local execution
        # So we force cloud execution
        elastic.elastic_action.actions = [
            elastic.elastic_action.actions[2]  # Full cloud action
        ]
        
        with pytest.raises(RuntimeError):
            elastic.elastic_execute("fail_release", "test")
            
    def test_adaptive_data_transfer(self, data_transfer):
        """Test adaptive data transfer based on bandwidth"""
        # Test with different bandwidth conditions
        bandwidths = [0.5, 5.0, 50.0]  # Mbps
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        for bandwidth in bandwidths:
            optimized = data_transfer.optimize_for_bandwidth(image, bandwidth)
            
            # Verify optimization
            assert 'bandwidth_mbps' in optimized
            assert optimized['bandwidth_mbps'] == bandwidth
            
            # Higher compression for lower bandwidth
            if bandwidth < 1.0:
                assert data_transfer.compression_level >= 7
            elif bandwidth > 10.0:
                assert data_transfer.compression_level <= 6
                
    def test_streaming_data_transfer(self):
        """Test streaming data transfer for large datasets"""
        from elasticros_core.utils import StreamingDataTransfer
        
        streamer = StreamingDataTransfer(chunk_size=65536)
        
        # Create large data
        large_data = np.random.randn(1000, 1000).astype(np.float32)
        
        # Prepare for streaming
        stream_info = streamer.prepare_stream(large_data)
        
        assert 'stream_id' in stream_info
        assert 'num_chunks' in stream_info
        assert stream_info['num_chunks'] > 1
        
        # Test chunk retrieval
        reconstructed = []
        for i in range(stream_info['num_chunks']):
            chunk = streamer.get_chunk(stream_info['stream_id'], i)
            assert chunk is not None
            reconstructed.append(chunk)
            
        # Verify reconstruction
        reconstructed_data = b''.join(reconstructed)
        assert len(reconstructed_data) == stream_info['total_size']
        
        # Clean up
        streamer.close_stream(stream_info['stream_id'])
        
    def test_node_state_consistency(self, mock_press_node, mock_release_node):
        """Test that node states remain consistent"""
        elastic = ElasticNode()
        elastic.register_node_pair("test", mock_press_node, mock_release_node)
        
        # Execute multiple times
        for i in range(10):
            result = elastic.elastic_execute("test", f"data_{i}")
            
        # Check node states
        press_stats = mock_press_node.get_stats()
        release_stats = mock_release_node.get_stats()
        
        # Verify execution counts
        total_executions = press_stats['execution_count'] + release_stats['execution_count']
        assert total_executions == 10
        
        # Verify timing consistency
        assert press_stats['average_execution_time'] >= 0
        assert release_stats['average_execution_time'] >= 0
        
    def test_partial_computation_handoff(self):
        """Test handoff between partial local and cloud computation"""
        class PartialPressNode(PressNode):
            def _initialize(self):
                self.stages = ['preprocess', 'feature_extract', 'transform']
                
            def _process(self, data, compute_ratio):
                completed_stages = int(compute_ratio * len(self.stages))
                
                result = {'original_data': data, 'completed_stages': []}
                
                for i in range(completed_stages):
                    stage = self.stages[i]
                    # Simulate processing
                    result[stage] = f"{stage}_done"
                    result['completed_stages'].append(stage)
                    
                result['remaining_stages'] = self.stages[completed_stages:]
                return result
                
        class PartialReleaseNode(ReleaseNode):
            def _initialize(self):
                pass
                
            def _process(self, data, compute_ratio):
                # Complete remaining stages
                result = data.copy() if isinstance(data, dict) else {'data': data}
                
                if 'remaining_stages' in data:
                    for stage in data['remaining_stages']:
                        result[stage] = f"{stage}_done_cloud"
                        
                result['cloud_processed'] = True
                return result
                
        elastic = ElasticNode()
        elastic.register_node_pair(
            "partial",
            PartialPressNode("partial_press"),
            PartialReleaseNode("partial_release")
        )
        
        # Force partial computation (50/50 split)
        elastic.elastic_action.actions = [
            elastic.elastic_action.Action(id=1, press_ratio=0.5, release_ratio=0.5)
        ]
        
        result = elastic.elastic_execute("partial", "test_data")
        
        # Verify partial processing occurred
        assert 'completed_stages' in result
        assert 'cloud_processed' in result
        assert len(result['completed_stages']) > 0
        assert len(result['completed_stages']) < 3  # Not all stages done locally
        
    @pytest.mark.parametrize("data_type,data", [
        ("image", np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)),
        ("audio", np.random.randn(16000).astype(np.float32)),
        ("pointcloud", np.random.randn(1000, 3).astype(np.float32)),
        ("json", {"sensor": "lidar", "timestamp": 12345, "values": [1, 2, 3]}),
        ("string", "Hello ElasticROS with special chars: 你好 🤖"),
    ])
    def test_different_data_types(self, data_type, data, data_transfer):
        """Test communication with different data types"""
        # Serialize
        packed = data_transfer.prepare_for_cloud(data)
        
        # Deserialize
        unpacked = data_transfer.receive_from_cloud(packed)
        
        # Verify based on type
        if isinstance(data, np.ndarray):
            np.testing.assert_array_almost_equal(data, unpacked)
        elif isinstance(data, (dict, list)):
            assert data == unpacked
        elif isinstance(data, str):
            assert data == unpacked
        else:
            assert data == pytest.approx(unpacked)


class TestNetworkSimulation:
    """Test cases for network condition simulation"""
    
    def test_bandwidth_variation(self):
        """Test node behavior under varying bandwidth"""
        from elasticros_core.utils import NetworkMonitor
        
        # Mock bandwidth changes
        monitor = NetworkMonitor()
        
        bandwidths = [100, 50, 10, 5, 1, 10, 50, 100]  # Mbps
        
        with patch.object(monitor, 'get_current_bandwidth') as mock_bandwidth:
            for bw in bandwidths:
                mock_bandwidth.return_value = bw
                
                # Verify bandwidth reading
                assert monitor.get_current_bandwidth() == bw
                
                # Estimate transfer time for 10MB
                transfer_time = monitor.estimate_transfer_time(10 * 1024 * 1024)
                
                # Lower bandwidth should mean longer transfer
                expected_time = (10 * 8) / bw  # Convert MB to Mb
                assert transfer_time >= expected_time  # Include latency
                
    def test_latency_impact(self):
        """Test impact of network latency"""
        from elasticros_core.utils import NetworkMonitor
        
        monitor = NetworkMonitor()
        
        latencies = [10, 50, 100, 200, 500]  # ms
        
        with patch.object(monitor, 'get_latency') as mock_latency:
            for latency in latencies:
                mock_latency.return_value = latency
                
                # Small data should be dominated by latency
                small_data_time = monitor.estimate_transfer_time(1024)  # 1KB
                assert small_data_time >= latency / 1000  # Convert ms to s
                
    def test_packet_loss_simulation(self):
        """Test behavior under packet loss"""
        # This would require more sophisticated network simulation
        # For now, test that system continues to function
        
        class UnreliableReleaseNode(ReleaseNode):
            def __init__(self, *args, packet_loss=0.1, **kwargs):
                self.packet_loss = packet_loss
                super().__init__(*args, **kwargs)
                
            def _initialize(self):
                pass
                
            def _process(self, data, compute_ratio):
                # Simulate packet loss
                import random
                if random.random() < self.packet_loss:
                    raise ConnectionError("Simulated packet loss")
                    
                return {"result": "success", "data": data}
                
        # Test with retries
        node = UnreliableReleaseNode("unreliable", packet_loss=0.3)
        
        success_count = 0
        failure_count = 0
        
        for i in range(100):
            try:
                result = node.compute(f"data_{i}")
                success_count += 1
            except ConnectionError:
                failure_count += 1
                
        # Should have some failures but not all
        assert failure_count > 0
        assert success_count > 0
        assert 0.2 < failure_count / 100 < 0.4  # Around 30% loss


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
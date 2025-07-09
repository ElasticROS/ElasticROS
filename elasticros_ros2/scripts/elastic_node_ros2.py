#!/usr/bin/env python3
"""
ROS2 wrapper for ElasticNode - integrates ElasticROS with ROS2
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
import numpy as np
import yaml
import threading
from typing import Dict, Any, Callable

# Import ElasticROS core
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from elasticros_core import ElasticNode, PressNode, ReleaseNode
from elasticros_core.utils import DataTransfer


class ROS2PressNode(PressNode):
    """Base class for ROS2-integrated Press Nodes"""
    
    def __init__(self, node_name: str, ros2_node: Node, config: Dict = None):
        super().__init__(node_name, config)
        self.ros2_node = ros2_node
        
        # Publishers and subscriptions will be set up by subclasses
        self.publishers = {}
        self.subscriptions = {}
        
    def create_publisher(self, topic: str, msg_type: type, qos_profile: QoSProfile = None):
        """Create ROS2 publisher"""
        if qos_profile is None:
            qos_profile = QoSProfile(depth=10)
            
        self.publishers[topic] = self.ros2_node.create_publisher(
            msg_type, topic, qos_profile
        )
        
    def create_subscription(self, topic: str, msg_type: type, callback: Callable, 
                           qos_profile: QoSProfile = None):
        """Create ROS2 subscription"""
        if qos_profile is None:
            qos_profile = QoSProfile(depth=10)
            
        self.subscriptions[topic] = self.ros2_node.create_subscription(
            msg_type, topic, callback, qos_profile
        )


class ElasticNodeROS2(Node):
    """ROS2 node wrapper for ElasticNode"""
    
    def __init__(self):
        super().__init__('elastic_node')
        
        # Declare parameters
        self.declare_parameter('config_file', '')
        self.declare_parameter('optimization_metric', 'latency')
        self.declare_parameter('publish_rate', 1.0)
        
        # Get parameters
        config_file = self.get_parameter('config_file').value
        
        # Initialize ElasticNode
        self.elastic_node = ElasticNode(config_file if config_file else None)
        
        # Data transfer utility
        self.data_transfer = DataTransfer()
        
        # Track registered nodes
        self.ros2_press_nodes = {}
        self.ros2_release_nodes = {}
        
        # Publishers
        self.status_pub = self.create_publisher(String, '~/status', 10)
        self.performance_pub = self.create_publisher(String, '~/performance', 10)
        
        # Create timer for status publishing
        publish_rate = self.get_parameter('publish_rate').value
        self.status_timer = self.create_timer(1.0 / publish_rate, self.publish_status)
        
        # Register example nodes
        self._register_example_nodes()
        
        self.get_logger().info('ElasticNode ROS2 wrapper initialized')
        
    def _register_example_nodes(self):
        """Register example node pairs"""
        # Image processing example
        image_press = ImagePressNodeROS2("image_press", self)
        image_release = ImageReleaseNodeROS2("image_release")
        
        self.elastic_node.register_node_pair(
            "image_processing",
            image_press,
            image_release
        )
        
        self.ros2_press_nodes["image_processing"] = image_press
        self.ros2_release_nodes["image_processing"] = image_release
        
    def publish_status(self):
        """Publish elastic node status"""
        stats = self.elastic_node.get_statistics()
        
        status_msg = String()
        status_msg.data = yaml.dump(stats)
        self.status_pub.publish(status_msg)
        
        # Also publish performance metrics
        if stats.get('average_execution_time') is not None:
            perf_msg = String()
            perf_msg.data = f"avg_time: {stats['average_execution_time']*1000:.1f}ms"
            self.performance_pub.publish(perf_msg)


class ImagePressNodeROS2(ROS2PressNode):
    """ROS2 Press Node for image processing"""
    
    def _initialize(self):
        """Initialize image processing resources"""
        # Set up ROS2 communication
        self.create_subscription(
            'image_raw', Image, self.image_callback
        )
        
        self.create_publisher(
            'image_processed', Image
        )
        
        # Image processing parameters
        self.target_size = self.config.get('target_size', (640, 480))
        
        self.ros2_node.get_logger().info('ImagePressNodeROS2 initialized')
        
    def image_callback(self, msg: Image):
        """Handle incoming images"""
        # Convert ROS2 image to numpy
        image = self.ros2_image_to_numpy(msg)
        
        # Process with elastic computing
        result = self.ros2_node.elastic_node.elastic_execute("image_processing", image)
        
        # Publish result if it contains processed image
        if isinstance(result, dict) and 'processed_image' in result:
            out_msg = self.numpy_to_ros2_image(result['processed_image'])
            self.publishers['image_processed'].publish(out_msg)
            
    def _process(self, data: np.ndarray, compute_ratio: float) -> Dict:
        """Process image based on compute ratio"""
        # Simple preprocessing example
        if compute_ratio == 0.0:
            return {'raw_image': data}
            
        # Resize
        if data.shape[:2] != self.target_size:
            import cv2
            data = cv2.resize(data, self.target_size)
            
        if compute_ratio < 0.5:
            return {'processed_image': data, 'level': 'minimal'}
            
        # Additional preprocessing for higher ratios
        # Normalize, extract features, etc.
        normalized = data.astype(np.float32) / 255.0
        
        return {
            'processed_image': normalized,
            'level': 'full',
            'preprocessing_done': True
        }
        
    @staticmethod
    def ros2_image_to_numpy(msg: Image) -> np.ndarray:
        """Convert ROS2 Image to numpy array"""
        if msg.encoding == 'rgb8':
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3
            )
        elif msg.encoding == 'mono8':
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width
            )
        else:
            raise ValueError(f"Unsupported encoding: {msg.encoding}")
            
    @staticmethod
    def numpy_to_ros2_image(arr: np.ndarray) -> Image:
        """Convert numpy array to ROS2 Image"""
        msg = Image()
        msg.height = arr.shape[0]
        msg.width = arr.shape[1]
        
        if len(arr.shape) == 3:
            msg.encoding = 'rgb8'
            msg.step = msg.width * 3
        else:
            msg.encoding = 'mono8'
            msg.step = msg.width
            
        msg.data = arr.tobytes()
        return msg


class ImageReleaseNodeROS2(ReleaseNode):
    """ROS2 Release Node for image processing"""
    
    def _initialize(self):
        """Initialize cloud processing resources"""
        # Simulate model loading
        self.ros2_node.get_logger().info('Loading cloud model...')
        import time
        time.sleep(0.5)
        self.model_loaded = True
        
    def _process(self, data: Dict, compute_ratio: float) -> Dict:
        """Process image in cloud"""
        # Simple mock processing
        import time
        time.sleep(0.1)  # Simulate processing time
        
        return {
            'detections': [
                {'class': 'object', 'confidence': 0.95},
                {'class': 'object', 'confidence': 0.87}
            ],
            'processed_image': data.get('processed_image', data.get('raw_image')),
            'cloud_processed': True
        }


class ExampleSubscriberNode(Node):
    """Example node that uses ElasticROS for processing"""
    
    def __init__(self):
        super().__init__('example_subscriber')
        
        # Subscribe to camera
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # Publisher for results
        self.result_pub = self.create_publisher(String, 'detection_results', 10)
        
        self.get_logger().info('Example subscriber node started')
        
    def image_callback(self, msg: Image):
        """Process incoming images with ElasticROS"""
        # This would interface with the ElasticNode
        # For now, just log
        self.get_logger().info(f'Received image: {msg.width}x{msg.height}')
        
        # Publish mock result
        result = String()
        result.data = f'Processed image at {self.get_clock().now()}'
        self.result_pub.publish(result)


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    # Create nodes
    elastic_node = ElasticNodeROS2()
    
    # Create executor for multi-threaded execution
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(elastic_node)
    
    try:
        # Spin
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        elastic_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
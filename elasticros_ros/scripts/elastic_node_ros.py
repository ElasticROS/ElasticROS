#!/usr/bin/env python3
"""
ROS wrapper for ElasticNode - integrates ElasticROS with ROS1
"""

import rospy
from std_msgs.msg import String, Float32, Header
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
import numpy as np
import yaml
import threading
from typing import Dict, Any, Callable

# Import ElasticROS core
from elasticros_core import ElasticNode, PressNode, ReleaseNode
from elasticros_core.utils import DataTransfer


class ROSPressNode(PressNode):
    """Base class for ROS-integrated Press Nodes"""
    
    def __init__(self, node_name: str, config: Dict = None):
        super().__init__(node_name, config)
        
        # ROS publishers/subscribers will be set up by subclasses
        self.publishers = {}
        self.subscribers = {}
        
    def create_publisher(self, topic: str, msg_type: type, queue_size: int = 10):
        """Create ROS publisher"""
        self.publishers[topic] = rospy.Publisher(topic, msg_type, queue_size=queue_size)
        
    def create_subscriber(self, topic: str, msg_type: type, callback: Callable):
        """Create ROS subscriber"""
        self.subscribers[topic] = rospy.Subscriber(topic, msg_type, callback)


class ROSReleaseNode(ReleaseNode):
    """Base class for ROS-integrated Release Nodes"""
    
    def __init__(self, node_name: str, config: Dict = None):
        super().__init__(node_name, config)
        
        # Will be initialized when deployed to cloud
        self.ros_node_initialized = False
        
    def initialize_ros_node(self):
        """Initialize ROS node in cloud instance"""
        if not self.ros_node_initialized:
            # This would be called on the cloud instance
            rospy.init_node(f"{self.node_name}_cloud", anonymous=True)
            self.ros_node_initialized = True


class ElasticNodeROS:
    """ROS wrapper for ElasticNode"""
    
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('elastic_node', anonymous=False)
        
        # Get parameters
        config_file = rospy.get_param('~config_file', None)
        
        # Initialize ElasticNode
        self.elastic_node = ElasticNode(config_file)
        
        # Data transfer utility
        self.data_transfer = DataTransfer()
        
        # Track registered ROS nodes
        self.ros_press_nodes = {}
        self.ros_release_nodes = {}
        
        # Publishers
        self.status_pub = rospy.Publisher(
            '~status', String, queue_size=1
        )
        self.performance_pub = rospy.Publisher(
            '~performance', String, queue_size=1
        )
        
        # Service servers
        # self.compute_service = rospy.Service(
        #     '~compute', ElasticCompute, self.handle_compute_request
        # )
        
        # Start status publishing
        self.status_timer = rospy.Timer(
            rospy.Duration(1.0), self.publish_status
        )
        
        rospy.loginfo("ElasticNode ROS wrapper initialized")
        
    def register_ros_nodes(self):
        """Register ROS-specific node pairs from parameters"""
        # Get node configurations from parameter server
        press_nodes = rospy.get_param('~press_nodes', {})
        release_nodes = rospy.get_param('~release_nodes', {})
        
        for node_name, config in press_nodes.items():
            # Create press node based on type
            node_type = config.get('type', 'generic')
            
            if node_type == 'image':
                press_node = ImagePressNodeROS(node_name, config)
            elif node_type == 'speech':
                press_node = SpeechPressNodeROS(node_name, config)
            else:
                press_node = GenericPressNodeROS(node_name, config)
                
            self.ros_press_nodes[node_name] = press_node
            
            # Find corresponding release node
            if node_name in release_nodes:
                release_config = release_nodes[node_name]
                
                if node_type == 'image':
                    release_node = ImageReleaseNodeROS(node_name, release_config)
                elif node_type == 'speech':
                    release_node = SpeechReleaseNodeROS(node_name, release_config)
                else:
                    release_node = GenericReleaseNodeROS(node_name, release_config)
                    
                self.ros_release_nodes[node_name] = release_node
                
                # Register with elastic node
                self.elastic_node.register_node_pair(
                    node_name, press_node, release_node
                )
                
                rospy.loginfo(f"Registered node pair: {node_name}")
                
    def publish_status(self, event):
        """Publish elastic node status"""
        stats = self.elastic_node.get_statistics()
        
        status_msg = String()
        status_msg.data = yaml.dump(stats)
        self.status_pub.publish(status_msg)
        
    def spin(self):
        """Main loop"""
        rospy.loginfo("ElasticNode ROS wrapper running")
        
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down ElasticNode")
        finally:
            self.elastic_node.shutdown()


class ImagePressNodeROS(ROSPressNode):
    """ROS Press Node for image processing"""
    
    def _initialize(self):
        """Initialize image processing resources"""
        # Set up ROS communication
        self.create_subscriber(
            'image_raw', Image, self.image_callback
        )
        self.create_publisher(
            'image_processed', Image
        )
        
        # Image processing parameters
        self.target_size = self.config.get('target_size', (640, 480))
        
    def image_callback(self, msg: Image):
        """Handle incoming images"""
        # Convert ROS image to numpy
        image = self.ros_image_to_numpy(msg)
        
        # Process with elastic computing
        result = self.elastic_node.elastic_execute(self.node_name, image)
        
        # Publish result
        if 'processed_image' in result:
            out_msg = self.numpy_to_ros_image(result['processed_image'], msg.header)
            self.publishers['image_processed'].publish(out_msg)
            
    def _process(self, data: np.ndarray, compute_ratio: float) -> Dict:
        """Process image based on compute ratio"""
        # Implement image preprocessing
        # Similar to the example implementation
        pass
        
    @staticmethod
    def ros_image_to_numpy(msg: Image) -> np.ndarray:
        """Convert ROS Image to numpy array"""
        # Simple implementation - in practice use cv_bridge
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
    def numpy_to_ros_image(arr: np.ndarray, header: Header) -> Image:
        """Convert numpy array to ROS Image"""
        msg = Image()
        msg.header = header
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


class GenericPressNodeROS(ROSPressNode):
    """Generic ROS Press Node for any data type"""
    
    def _initialize(self):
        """Initialize based on configuration"""
        # Set up topics from config
        input_topic = self.config.get('input_topic', 'input')
        output_topic = self.config.get('output_topic', 'output')
        
        # For now, assume String messages
        self.create_subscriber(
            input_topic, String, self.data_callback
        )
        self.create_publisher(
            output_topic, String
        )
        
    def data_callback(self, msg: String):
        """Handle incoming data"""
        # Process with elastic computing
        result = self.elastic_node.elastic_execute(self.node_name, msg.data)
        
        # Publish result
        out_msg = String()
        out_msg.data = str(result)
        self.publishers[self.config.get('output_topic', 'output')].publish(out_msg)
        
    def _process(self, data: Any, compute_ratio: float) -> Any:
        """Generic processing"""
        # Simple pass-through with simulated delay
        import time
        time.sleep(0.1 * compute_ratio)
        
        return {
            'processed_data': f"Processed({compute_ratio:.1%}): {data}",
            'compute_ratio': compute_ratio
        }


def main():
    """Main entry point"""
    try:
        # Create and run the ROS wrapper
        wrapper = ElasticNodeROS()
        wrapper.register_ros_nodes()
        wrapper.spin()
        
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in ElasticNode ROS wrapper: {e}")
        raise


if __name__ == '__main__':
    main()
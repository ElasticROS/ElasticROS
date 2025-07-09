#!/usr/bin/env python3
"""
ROS communication wrapper for ElasticROS
Handles ROS-specific message conversion and topic management
"""

import rospy
from std_msgs.msg import String, Header, Float32, Int32
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from geometry_msgs.msg import PoseStamped, Twist
import numpy as np
from typing import Any, Dict, Union, Callable
import json
import base64

# Try to import cv_bridge for better image handling
try:
    from cv_bridge import CvBridge
    CV_BRIDGE_AVAILABLE = True
except ImportError:
    CV_BRIDGE_AVAILABLE = False
    rospy.logwarn("cv_bridge not available, using basic image conversion")


class ROSDataConverter:
    """Converts between ROS messages and ElasticROS data formats"""
    
    def __init__(self):
        if CV_BRIDGE_AVAILABLE:
            self.bridge = CvBridge()
        else:
            self.bridge = None
            
    def ros_to_elastic(self, msg: Any) -> Dict[str, Any]:
        """Convert ROS message to ElasticROS format"""
        msg_type = type(msg).__name__
        
        if isinstance(msg, Image):
            return self._image_to_dict(msg)
        elif isinstance(msg, CompressedImage):
            return self._compressed_image_to_dict(msg)
        elif isinstance(msg, PointCloud2):
            return self._pointcloud_to_dict(msg)
        elif isinstance(msg, String):
            return {'data': msg.data, 'type': 'string'}
        elif isinstance(msg, (Float32, Int32)):
            return {'data': msg.data, 'type': msg_type.lower()}
        elif isinstance(msg, PoseStamped):
            return self._pose_to_dict(msg)
        elif isinstance(msg, Twist):
            return self._twist_to_dict(msg)
        else:
            # Generic message - convert to dict
            return self._generic_msg_to_dict(msg)
            
    def elastic_to_ros(self, data: Dict[str, Any], msg_class: type) -> Any:
        """Convert ElasticROS data to ROS message"""
        if msg_class == Image:
            return self._dict_to_image(data)
        elif msg_class == CompressedImage:
            return self._dict_to_compressed_image(data)
        elif msg_class == String:
            return String(data=str(data.get('data', '')))
        elif msg_class == Float32:
            return Float32(data=float(data.get('data', 0.0)))
        elif msg_class == Int32:
            return Int32(data=int(data.get('data', 0)))
        else:
            # Try to construct message from dict
            return self._dict_to_generic_msg(data, msg_class)
            
    def _image_to_dict(self, msg: Image) -> Dict[str, Any]:
        """Convert ROS Image to dictionary"""
        if self.bridge:
            # Use cv_bridge for proper conversion
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                return {
                    'data': cv_image,
                    'type': 'image',
                    'encoding': msg.encoding,
                    'width': msg.width,
                    'height': msg.height,
                    'header': self._header_to_dict(msg.header)
                }
            except Exception as e:
                rospy.logwarn(f"cv_bridge conversion failed: {e}")
                
        # Fallback to basic conversion
        data = np.frombuffer(msg.data, dtype=np.uint8)
        
        if msg.encoding == 'rgb8':
            data = data.reshape((msg.height, msg.width, 3))
        elif msg.encoding == 'bgr8':
            data = data.reshape((msg.height, msg.width, 3))
            data = data[:, :, ::-1]  # BGR to RGB
        elif msg.encoding == 'mono8':
            data = data.reshape((msg.height, msg.width))
        else:
            rospy.logwarn(f"Unsupported encoding: {msg.encoding}")
            
        return {
            'data': data,
            'type': 'image',
            'encoding': msg.encoding,
            'width': msg.width,
            'height': msg.height,
            'header': self._header_to_dict(msg.header)
        }
        
    def _dict_to_image(self, data: Dict[str, Any]) -> Image:
        """Convert dictionary to ROS Image"""
        msg = Image()
        
        if 'header' in data:
            msg.header = self._dict_to_header(data['header'])
        else:
            msg.header.stamp = rospy.Time.now()
            
        image_data = data.get('data')
        
        if isinstance(image_data, np.ndarray):
            msg.height = image_data.shape[0]
            msg.width = image_data.shape[1]
            
            if len(image_data.shape) == 3:
                msg.encoding = data.get('encoding', 'rgb8')
                msg.step = msg.width * 3
            else:
                msg.encoding = 'mono8'
                msg.step = msg.width
                
            msg.data = image_data.tobytes()
        else:
            # Assume raw bytes
            msg.data = image_data
            msg.height = data.get('height', 480)
            msg.width = data.get('width', 640)
            msg.encoding = data.get('encoding', 'rgb8')
            msg.step = msg.width * (3 if msg.encoding in ['rgb8', 'bgr8'] else 1)
            
        return msg
        
    def _compressed_image_to_dict(self, msg: CompressedImage) -> Dict[str, Any]:
        """Convert compressed image to dict"""
        return {
            'data': base64.b64encode(msg.data).decode('utf-8'),
            'type': 'compressed_image',
            'format': msg.format,
            'header': self._header_to_dict(msg.header)
        }
        
    def _dict_to_compressed_image(self, data: Dict[str, Any]) -> CompressedImage:
        """Convert dict to compressed image"""
        msg = CompressedImage()
        msg.header = self._dict_to_header(data.get('header', {}))
        msg.format = data.get('format', 'jpeg')
        msg.data = base64.b64decode(data['data'])
        return msg
        
    def _pointcloud_to_dict(self, msg: PointCloud2) -> Dict[str, Any]:
        """Convert PointCloud2 to dictionary"""
        # Simplified - in practice would use ros_numpy or similar
        return {
            'type': 'pointcloud2',
            'width': msg.width,
            'height': msg.height,
            'point_step': msg.point_step,
            'row_step': msg.row_step,
            'data': base64.b64encode(msg.data).decode('utf-8'),
            'fields': [{'name': f.name, 'offset': f.offset, 'datatype': f.datatype, 'count': f.count} 
                      for f in msg.fields],
            'header': self._header_to_dict(msg.header)
        }
        
    def _pose_to_dict(self, msg: PoseStamped) -> Dict[str, Any]:
        """Convert PoseStamped to dictionary"""
        return {
            'type': 'pose_stamped',
            'header': self._header_to_dict(msg.header),
            'position': {
                'x': msg.pose.position.x,
                'y': msg.pose.position.y,
                'z': msg.pose.position.z
            },
            'orientation': {
                'x': msg.pose.orientation.x,
                'y': msg.pose.orientation.y,
                'z': msg.pose.orientation.z,
                'w': msg.pose.orientation.w
            }
        }
        
    def _twist_to_dict(self, msg: Twist) -> Dict[str, Any]:
        """Convert Twist to dictionary"""
        return {
            'type': 'twist',
            'linear': {
                'x': msg.linear.x,
                'y': msg.linear.y,
                'z': msg.linear.z
            },
            'angular': {
                'x': msg.angular.x,
                'y': msg.angular.y,
                'z': msg.angular.z
            }
        }
        
    def _header_to_dict(self, header: Header) -> Dict[str, Any]:
        """Convert Header to dictionary"""
        return {
            'seq': header.seq,
            'stamp': {'secs': header.stamp.secs, 'nsecs': header.stamp.nsecs},
            'frame_id': header.frame_id
        }
        
    def _dict_to_header(self, data: Dict[str, Any]) -> Header:
        """Convert dictionary to Header"""
        header = Header()
        header.seq = data.get('seq', 0)
        
        if 'stamp' in data:
            header.stamp = rospy.Time(data['stamp']['secs'], data['stamp']['nsecs'])
        else:
            header.stamp = rospy.Time.now()
            
        header.frame_id = data.get('frame_id', '')
        return header
        
    def _generic_msg_to_dict(self, msg: Any) -> Dict[str, Any]:
        """Convert generic ROS message to dictionary"""
        result = {'type': type(msg).__name__}
        
        # Get all message fields
        for slot in msg.__slots__:
            attr = getattr(msg, slot)
            
            if hasattr(attr, '__slots__'):
                # Nested message
                result[slot] = self._generic_msg_to_dict(attr)
            elif isinstance(attr, (list, tuple)):
                # Array field
                result[slot] = list(attr)
            else:
                # Simple field
                result[slot] = attr
                
        return result
        
    def _dict_to_generic_msg(self, data: Dict[str, Any], msg_class: type) -> Any:
        """Convert dictionary to generic ROS message"""
        msg = msg_class()
        
        for key, value in data.items():
            if key == 'type':
                continue
                
            if hasattr(msg, key):
                setattr(msg, key, value)
                
        return msg


class TopicManager:
    """Manages ROS topic subscriptions and publications for ElasticROS"""
    
    def __init__(self):
        self.converter = ROSDataConverter()
        self.publishers = {}
        self.subscribers = {}
        self.callbacks = {}
        
    def create_publisher(self, topic: str, msg_type: type, queue_size: int = 10):
        """Create a ROS publisher"""
        if topic not in self.publishers:
            self.publishers[topic] = rospy.Publisher(topic, msg_type, queue_size=queue_size)
            rospy.loginfo(f"Created publisher for topic: {topic}")
            
    def create_subscriber(self, topic: str, msg_type: type, callback: Callable):
        """Create a ROS subscriber with ElasticROS data conversion"""
        def wrapper_callback(msg):
            # Convert ROS message to ElasticROS format
            elastic_data = self.converter.ros_to_elastic(msg)
            # Call user callback with converted data
            callback(elastic_data)
            
        self.subscribers[topic] = rospy.Subscriber(topic, msg_type, wrapper_callback)
        self.callbacks[topic] = callback
        rospy.loginfo(f"Created subscriber for topic: {topic}")
        
    def publish(self, topic: str, data: Union[Dict, Any], msg_type: type = None):
        """Publish data to ROS topic"""
        if topic not in self.publishers:
            rospy.logwarn(f"No publisher for topic: {topic}")
            return
            
        publisher = self.publishers[topic]
        
        # Convert data if needed
        if isinstance(data, dict) and msg_type:
            msg = self.converter.elastic_to_ros(data, msg_type)
        else:
            msg = data
            
        publisher.publish(msg)
        
    def unregister(self, topic: str):
        """Unregister publisher or subscriber"""
        if topic in self.publishers:
            self.publishers[topic].unregister()
            del self.publishers[topic]
            
        if topic in self.subscribers:
            self.subscribers[topic].unregister()
            del self.subscribers[topic]
            
        if topic in self.callbacks:
            del self.callbacks[topic]


class ServiceWrapper:
    """Wraps ROS services for ElasticROS"""
    
    def __init__(self):
        self.converter = ROSDataConverter()
        self.services = {}
        self.clients = {}
        
    def create_service(self, service_name: str, service_type: type, handler: Callable):
        """Create ROS service with ElasticROS data conversion"""
        def wrapper_handler(req):
            # Convert request to ElasticROS format
            elastic_req = self.converter.ros_to_elastic(req)
            
            # Call handler
            elastic_resp = handler(elastic_req)
            
            # Convert response back to ROS
            resp_class = service_type._response_class
            return self.converter.elastic_to_ros(elastic_resp, resp_class)
            
        self.services[service_name] = rospy.Service(
            service_name, service_type, wrapper_handler
        )
        rospy.loginfo(f"Created service: {service_name}")
        
    def create_client(self, service_name: str, service_type: type):
        """Create ROS service client"""
        rospy.wait_for_service(service_name, timeout=5.0)
        self.clients[service_name] = rospy.ServiceProxy(service_name, service_type)
        rospy.loginfo(f"Created service client: {service_name}")
        
    def call_service(self, service_name: str, request_data: Dict) -> Dict:
        """Call ROS service with ElasticROS data"""
        if service_name not in self.clients:
            raise ValueError(f"No client for service: {service_name}")
            
        client = self.clients[service_name]
        
        # Get request type and convert data
        req_class = client._request_class
        req = self.converter.elastic_to_ros(request_data, req_class)
        
        # Call service
        try:
            resp = client(req)
            # Convert response to ElasticROS format
            return self.converter.ros_to_elastic(resp)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            raise


# Global instances for easy access
_topic_manager = None
_service_wrapper = None


def get_topic_manager() -> TopicManager:
    """Get global TopicManager instance"""
    global _topic_manager
    if _topic_manager is None:
        _topic_manager = TopicManager()
    return _topic_manager


def get_service_wrapper() -> ServiceWrapper:
    """Get global ServiceWrapper instance"""
    global _service_wrapper
    if _service_wrapper is None:
        _service_wrapper = ServiceWrapper()
    return _service_wrapper
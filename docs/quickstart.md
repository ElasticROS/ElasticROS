# ElasticROS Quick Start Guide

Get started with ElasticROS in 15 minutes!

## Prerequisites

Before starting, ensure you have:
- ✅ ElasticROS installed ([Installation Guide](installation.md))
- ✅ AWS credentials configured (`aws configure`)
- ✅ ROS environment sourced

## 5-Minute Example: Hello ElasticROS

### 1. Test Local Installation

```python
# test_elastic.py
from elasticros_core import ElasticNode, DummyPressNode, DummyReleaseNode

# Create elastic node
elastic = ElasticNode()

# Create and register nodes
press = DummyPressNode("hello_press")
release = DummyReleaseNode("hello_release")
elastic.register_node_pair("hello_task", press, release)

# Execute with elastic computing
result = elastic.elastic_execute("hello_task", "Hello ElasticROS!")
print(f"Result: {result}")

# Show statistics
stats = elastic.get_statistics()
print(f"Execution time: {stats['average_execution_time']*1000:.1f}ms")
```

Run it:
```bash
python3 test_elastic.py
```

### 2. Setup AWS Environment (One-time)

```bash
# Run setup wizard
python3 scripts/setup_aws.py

# This will:
# - Create VPC and subnets
# - Setup security groups
# - Create SSH key pair
# - Test instance launch
```

## Real Example: Adaptive Image Processing

### 1. Basic Image Processing

```python
# adaptive_vision.py
import numpy as np
from elasticros_core import (
    ElasticNode,
    ImageProcessingPressNode,
    ImageProcessingReleaseNode
)

# Setup elastic computing
elastic = ElasticNode("config/default_config.yaml")

# Create nodes
press = ImageProcessingPressNode("vision_press")
release = ImageProcessingReleaseNode(
    "vision_release",
    config={'instance_type': 't2.micro', 'use_gpu': False}
)

# Register pair
elastic.register_node_pair("vision", press, release)

# Process image
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
result = elastic.elastic_execute("vision", image)

print(f"Detections: {len(result['detections'])}")
print(f"Processing location: {result['preprocessing_level']}")
```

### 2. Test Adaptation

```bash
# Run grasping example with bandwidth simulation
cd examples/grasping

# Normal conditions
python3 grasp_detection_node.py

# Simulate low bandwidth (forces local processing)
python3 grasp_detection_node.py --simulate-bandwidth 2.0

# Use camera for real-time demo
python3 grasp_detection_node.py --camera
```

## ROS Integration Example

### 1. Launch ElasticROS

```bash
# Terminal 1: Start ROS core
roscore

# Terminal 2: Launch ElasticROS
roslaunch elasticros_ros elasticros.launch
```

### 2. Custom ROS Node

```python
#!/usr/bin/env python3
# elastic_image_node.py

import rospy
from sensor_msgs.msg import Image
from elasticros_ros.elastic_node_ros import ROSPressNode, ElasticNodeROS

class MyImageProcessor(ROSPressNode):
    def _initialize(self):
        self.create_subscriber('camera/image', Image, self.process_image)
        self.create_publisher('processed_image', Image)
        
    def process_image(self, msg):
        # Convert ROS image to numpy
        image = self.ros_image_to_numpy(msg)
        
        # Process with ElasticROS
        result = self.elastic_node.elastic_execute("image_task", image)
        
        # Publish result
        self.publishers['processed_image'].publish(result)

if __name__ == '__main__':
    rospy.init_node('elastic_image_processor')
    processor = MyImageProcessor("my_processor")
    rospy.spin()
```

## Creating Your Own Elastic Application

### Step 1: Define Your Task

```python
from elasticros_core import PressNode, ReleaseNode

class MyTaskPressNode(PressNode):
    def _initialize(self):
        # Setup local resources
        self.model = load_lightweight_model()
        
    def _process(self, data, compute_ratio):
        if compute_ratio > 0.7:
            # Heavy local processing
            features = self.model.extract_features(data)
            return {'features': features}
        else:
            # Minimal processing
            return {'raw_data': preprocess(data)}

class MyTaskReleaseNode(ReleaseNode):
    def _initialize(self):
        # Setup cloud resources
        self.model = load_heavy_model()
        
    def _process(self, data, compute_ratio):
        if 'features' in data:
            # Use preprocessed features
            result = self.model.predict(data['features'])
        else:
            # Full processing
            result = self.model.process(data['raw_data'])
        return result
```

### Step 2: Configure Action Space

```yaml
# my_config.yaml
elastic_node:
  optimization_metric: "latency"
  action_space:
    - press_ratio: 1.0
      release_ratio: 0.0
      description: "Full local - for offline mode"
      
    - press_ratio: 0.8
      release_ratio: 0.2  
      description: "Heavy local preprocessing"
      
    - press_ratio: 0.2
      release_ratio: 0.8
      description: "Minimal local, cloud inference"
```

### Step 3: Run Your Application

```python
# Create elastic system
elastic = ElasticNode("my_config.yaml")

# Register your nodes
elastic.register_node_pair(
    "my_task",
    MyTaskPressNode("press"),
    MyTaskReleaseNode("release", config={'instance_type': 'g4dn.xlarge'})
)

# Process data adaptively
while True:
    data = get_sensor_data()
    result = elastic.elastic_execute("my_task", data)
    use_result(result)
```

## Monitoring and Debugging

### 1. Real-time Monitoring

```bash
# Monitor network
elasticros-monitor-network

# Monitor system metrics
elasticros-monitor --metrics cpu,memory,network
```

### 2. View Logs

```python
import logging

# Enable debug logging
logging.getLogger('elasticros_core').setLevel(logging.DEBUG)

# Log to file
logging.basicConfig(
    filename='elasticros.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### 3. Performance Analysis

```python
# Get detailed statistics
stats = elastic.get_statistics()
print(f"Average latency: {stats['average_execution_time']*1000:.1f}ms")
print(f"Action distribution: {stats['action_distribution']}")
print(f"Regret bound: {stats['regret_bound']}")

# Export metrics
elastic.export_metrics("performance_data.csv")
```

## Common Patterns

### 1. Bandwidth-Aware Processing

```python
# Automatically adapt to network conditions
elastic_node.optimization_metric = "latency"

# System will automatically:
# - Use cloud when bandwidth > 10 Mbps
# - Switch to local when bandwidth drops
# - Balance based on current conditions
```

### 2. CPU-Aware Offloading

```python
# Configure CPU thresholds
config = {
    'thresholds': {
        'max_cpu_usage': 70,  # Offload when CPU > 70%
        'max_memory_usage': 80
    }
}
```

### 3. Multi-Stage Pipeline

```python
# Chain multiple elastic computations
result1 = elastic.elastic_execute("detection", image)
result2 = elastic.elastic_execute("recognition", result1)
result3 = elastic.elastic_execute("planning", result2)
```

## Best Practices

1. **Start Simple**: Begin with provided examples before creating custom nodes

2. **Profile First**: Understand your computation bottlenecks
   ```python
   elastic.profile_mode = True
   result = elastic.elastic_execute("task", data)
   print(elastic.get_profile_report())
   ```

3. **Configure Action Space**: Design actions that match your network conditions
   - High bandwidth environment: Include more cloud-heavy options
   - Edge deployment: Focus on local computation options

4. **Monitor Costs**: Track AWS usage
   ```python
   costs = elastic.estimate_cloud_costs()
   print(f"Estimated hourly cost: ${costs['hourly']}")
   ```

5. **Handle Failures**: Implement fallbacks
   ```python
   try:
       result = elastic.elastic_execute("task", data)
   except CloudConnectionError:
       # Fallback to local only
       result = press_node.compute(data, 1.0)
   ```

## Next Steps

- 📖 Read the full [API Reference](api_reference.md)
- 🚀 Explore more [Examples](examples/)
- 🔧 Learn about [Advanced Configuration](advanced_config.md)
- 📊 Run [Performance Benchmarks](../examples/benchmarks/)

## Getting Help

- **Common Issues**: Check [Troubleshooting](installation.md#troubleshooting)
- **GitHub Issues**: https://github.com/ElasticROS/elasticros/issues
- **Community Forum**: https://forum.elasticros.org
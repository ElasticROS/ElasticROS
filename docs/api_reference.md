# ElasticROS API Reference

## Core Components

### ElasticNode

Central controller for elastic computing decisions.

```python
class ElasticNode(config_path: str = None)
```

#### Parameters
- `config_path` (str, optional): Path to YAML configuration file

#### Methods

##### `register_node_pair(function_name, press_node, release_node)`
Register a Press-Release node pair for elastic computing.

```python
elastic.register_node_pair("vision", press_node, release_node)
```

##### `elastic_execute(function_name, input_data) -> Any`
Execute function with automatic elastic computing decisions.

```python
result = elastic.elastic_execute("vision", image_data)
```

##### `get_statistics() -> Dict`
Get performance statistics.

```python
stats = elastic.get_statistics()
# Returns: {
#   'average_execution_time': 0.123,
#   'action_distribution': {0: 10, 1: 5, 2: 3},
#   'total_executions': 18,
#   'regret_bound': 25.3
# }
```

##### `shutdown()`
Clean shutdown of ElasticNode.

---

### ElasticAction

Online learning algorithm for adaptive resource allocation.

```python
class ElasticAction(
    action_space: List[Dict],
    context_dim: int,
    gamma: float = 1.0,
    beta_multiplier: float = 1.0,
    force_sample_rate: float = 0.1
)
```

#### Parameters
- `action_space`: List of possible actions with press/release ratios
- `context_dim`: Dimension of context vectors
- `gamma`: Regularization parameter (≥1)
- `beta_multiplier`: Confidence bound multiplier
- `force_sample_rate`: Rate for forced exploration

#### Methods

##### `select_action(context, data=None, robot_compute_cost=None) -> Action`
Select optimal action based on current context.

```python
action = elastic_action.select_action(
    context=np.array([0.5, 0.8, 0.2]),
    data=image_array
)
```

##### `update(action, context, reward)`
Update algorithm parameters based on observed reward.

##### `get_regret_bound() -> float`
Get theoretical regret bound at current time step.

---

### PressNode

Base class for local computation nodes.

```python
class PressNode(node_name: str, config: Dict = None)
```

#### Abstract Methods

##### `_initialize()`
Initialize node-specific resources. Must be implemented by subclasses.

##### `_process(data: Any, compute_ratio: float) -> Any`
Process data with given computation ratio. Must be implemented by subclasses.

#### Methods

##### `compute(data, compute_ratio=1.0) -> Any`
Execute computation with given ratio.

```python
result = press_node.compute(image, compute_ratio=0.7)
```

##### `get_stats() -> Dict`
Get node statistics.

---

### ReleaseNode

Base class for cloud computation nodes.

```python
class ReleaseNode(node_name: str, config: Dict = None)
```

#### Configuration Options
```python
config = {
    'instance_type': 't2.micro',  # EC2 instance type
    'use_gpu': False,             # Enable GPU acceleration
    'region': 'us-east-1'         # AWS region
}
```

#### Methods

Similar to PressNode, with additional:

##### `is_available() -> bool`
Check if cloud resources are available.

##### `get_estimated_cost(data_size: int) -> float`
Estimate cost for processing given data size.

---

## Utility Classes

### MetricsCollector

System metrics collection.

```python
from elasticros_core.utils import MetricsCollector

collector = MetricsCollector(history_size=100)
```

#### Methods

##### `get_cpu_usage() -> float`
Get current CPU usage percentage.

##### `get_memory_usage() -> float`
Get current memory usage percentage.

##### `get_power_consumption() -> float`
Get estimated power consumption in watts.

##### `get_system_stats() -> Dict`
Get comprehensive system statistics.

---

### NetworkMonitor

Network condition monitoring.

```python
from elasticros_core.utils import NetworkMonitor

monitor = NetworkMonitor(target_host="8.8.8.8")
```

#### Methods

##### `get_latency() -> float`
Get current network latency in milliseconds.

##### `get_current_bandwidth() -> float`
Get current bandwidth estimate in Mbps.

##### `estimate_transfer_time(data_size_bytes: int) -> float`
Estimate time to transfer data.

---

### DataTransfer

Efficient data serialization and compression.

```python
from elasticros_core.utils import DataTransfer

transfer = DataTransfer(compression_level=6)
```

#### Methods

##### `prepare_for_cloud(data) -> Dict`
Prepare data for cloud transfer with compression.

```python
packed = transfer.prepare_for_cloud(numpy_array)
# Returns: {
#   'data': 'base64_encoded_compressed_data',
#   'data_type': 'ndarray',
#   'original_size': 1048576,
#   'compressed_size': 102400
# }
```

##### `receive_from_cloud(transfer_dict) -> Any`
Reconstruct data from cloud transfer.

---

## Cloud Integration

### AWSManager

AWS EC2 instance management.

```python
from cloud_integration import AWSManager

aws = AWSManager(region='us-east-1')
```

#### Methods

##### `launch_instance(instance_name, instance_type, subnet_id, security_group_id) -> str`
Launch EC2 instance for cloud computation.

##### `terminate_instance(instance_name)`
Terminate EC2 instance.

##### `list_instances() -> List[Dict]`
List all ElasticROS instances.

---

### VPCManager

VPC configuration for robot-cloud communication.

```python
from cloud_integration import VPCManager

vpc = VPCManager(region='us-east-1')
```

#### Methods

##### `create_elasticros_vpc(name='ElasticROS-VPC') -> Dict`
Create complete VPC setup.

##### `cleanup_vpc(vpc_id)`
Clean up VPC and all associated resources.

---

## ROS Integration

### ElasticNodeROS

ROS wrapper for ElasticNode.

```python
from elasticros_ros import ElasticNodeROS

ros_elastic = ElasticNodeROS()
```

### ROSPressNode

Base class for ROS-integrated Press Nodes.

```python
class MyROSPressNode(ROSPressNode):
    def _initialize(self):
        self.create_subscriber('input_topic', MessageType, self.callback)
        self.create_publisher('output_topic', MessageType)
```

---

## Configuration

### YAML Configuration Structure

```yaml
elastic_node:
  optimization_metric: "latency"  # Options: latency, cpu, power, throughput
  context_dim: 10
  gamma: 1.0
  beta_multiplier: 1.0
  force_sample_rate: 0.1
  
  action_space:
    - press_ratio: 1.0
      release_ratio: 0.0
      description: "Full local computation"
      
    - press_ratio: 0.5
      release_ratio: 0.5
      description: "Balanced distribution"

network:
  latency_target: "8.8.8.8"
  bandwidth_test_interval: 300
  compression_level: 6

thresholds:
  max_latency: 1000        # ms
  min_bandwidth: 1.0       # Mbps
  max_cpu_usage: 80        # percent
  max_memory_usage: 90     # percent
```

---

## Examples

### Basic Usage

```python
from elasticros_core import ElasticNode, PressNode, ReleaseNode

# Define custom nodes
class MyPressNode(PressNode):
    def _initialize(self):
        self.processor = LocalProcessor()
        
    def _process(self, data, compute_ratio):
        if compute_ratio > 0.5:
            return self.processor.heavy_process(data)
        else:
            return self.processor.light_process(data)

class MyReleaseNode(ReleaseNode):
    def _initialize(self):
        self.model = CloudModel()
        
    def _process(self, data, compute_ratio):
        return self.model.infer(data)

# Setup elastic computing
elastic = ElasticNode()
elastic.register_node_pair(
    "my_task",
    MyPressNode("local"),
    MyReleaseNode("cloud")
)

# Execute adaptively
result = elastic.elastic_execute("my_task", input_data)
```

### Advanced Usage with Custom Context

```python
# Build custom context
def build_context():
    return np.array([
        metrics.get_cpu_usage() / 100,
        metrics.get_memory_usage() / 100,
        network.get_current_bandwidth() / 100,
        network.get_latency() / 1000,
        battery_level / 100,
        task_priority,
        data_size_mb / 10
    ])

# Custom decision making
context = build_context()
action = elastic.elastic_action.select_action(context, data)
result, time = elastic.execute_with_action("task", data, action)
```

### Error Handling

```python
try:
    result = elastic.elastic_execute("task", data)
except CloudConnectionError:
    # Fallback to local
    result = press_node.compute(data, 1.0)
except Exception as e:
    logger.error(f"Elastic execution failed: {e}")
    # Handle error
```

---

## Constants and Types

### Action
```python
@dataclass
class Action:
    id: int
    press_ratio: float    # [0.0, 1.0]
    release_ratio: float  # [0.0, 1.0]
    # Constraint: press_ratio + release_ratio == 1.0
```

### Optimization Metrics
- `"latency"`: Minimize end-to-end latency
- `"cpu"`: Minimize CPU usage
- `"power"`: Minimize power consumption  
- `"throughput"`: Maximize throughput

### Instance Types
Common AWS EC2 instance types:
- `"t2.micro"`: 1 vCPU, 1 GB RAM (free tier)
- `"t2.small"`: 1 vCPU, 2 GB RAM
- `"t2.medium"`: 2 vCPU, 4 GB RAM
- `"g4dn.xlarge"`: 4 vCPU, 16 GB RAM, 1 GPU

---

## Environment Variables

- `ELASTICROS_CONFIG`: Default configuration file path
- `ELASTICROS_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `AWS_REGION`: Default AWS region
- `ROS_IP`: ROS node IP address
- `ROS_MASTER_URI`: ROS master URI
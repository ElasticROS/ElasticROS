# ElasticROS

Algorithm-level elastic computing framework for Internet of Robotic Things (IoRT) based on ROS and ROS2.

## Overview

ElasticROS enables dynamic computational distribution between robots and cloud/edge servers through a novel Press-Elastic-Release node architecture. It uses the ElasticAction online learning algorithm to adaptively optimize resource allocation based on real-time conditions.

## Key Features

- **Algorithm-level elastic computing** - Fine-grained task distribution
- **Online learning optimization** - ElasticAction algorithm with proven O(T^0.75 log(T)) regret bound
- **Dual platform support** - Works with both ROS and ROS2
- **AWS cloud integration** - Automated EC2 deployment and VPC configuration
- **Real-time adaptation** - Responds to bandwidth and CPU changes dynamically

## Installation

### Prerequisites

- Ubuntu 20.04 (ROS Noetic) or Ubuntu 22.04 (ROS2 Humble)
- Python 3.8+
- AWS account with EC2 access
- CUDA 11.0+ (optional, for GPU acceleration)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/ElasticROS/elasticros.git
cd elasticros

# Install dependencies
./scripts/install_dependencies.sh

# Setup Python environment
pip install -r requirements.txt
python setup.py install

# Configure AWS credentials
aws configure

# Setup network environment
sudo ./scripts/network_setup.sh
```

## Quick Start

### 1. Configure ElasticROS

Edit `config/default_config.yaml`:

```yaml
elastic_node:
  optimization_metric: "latency"  # or "cpu", "power"
  action_space:
    - {press_ratio: 1.0, release_ratio: 0.0}  # Full local
    - {press_ratio: 0.7, release_ratio: 0.3}  # 70-30 split
    - {press_ratio: 0.3, release_ratio: 0.7}  # 30-70 split
    - {press_ratio: 0.0, release_ratio: 1.0}  # Full cloud
```

### 2. Launch ElasticROS

For ROS:
```bash
roslaunch elasticros_ros elasticros.launch
```

For ROS2:
```bash
ros2 launch elasticros_ros2 elasticros_launch.py
```

### 3. Run Example Application

```bash
# Grasping example
cd examples/grasping
python grasp_detection_node.py

# Human-robot dialogue example
cd examples/human_robot_dialogue
python speech_recognition_node.py
```

## Architecture

ElasticROS consists of three main node types:

- **Press Nodes**: Execute computations on the robot
- **Release Nodes**: Execute computations in the cloud
- **Elastic Node**: Controls the computational distribution using ElasticAction algorithm

## Examples

See the `examples/` directory for:
- Robotic grasping with dynamic computation offloading
- Human-robot dialogue with adaptive speech processing
- Performance benchmarks and comparisons with FogROS

## Documentation

Full documentation available at: https://github.com/ElasticROS/elasticros/wiki

## License

MIT License - see LICENSE file for details.

## Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

## Contact

- Issues: https://github.com/ElasticROS/elasticros/issues
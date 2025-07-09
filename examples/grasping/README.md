# ElasticROS Grasping Example

This example demonstrates how ElasticROS optimizes robotic grasping tasks by dynamically distributing computation between the robot and cloud.

## Overview

The grasping example implements:
- **Press Node**: Local image preprocessing (resize, normalize, edge detection)
- **Release Node**: Cloud-based deep learning inference for grasp detection
- **Elastic Node**: Adaptive decision making based on network conditions and CPU usage

## Requirements

- Python 3.8+
- OpenCV (`pip install opencv-python`)
- ElasticROS core installed
- (Optional) USB camera for live demo

## Quick Start

### 1. Basic Test

Run with simulated data:
```bash
python grasp_detection_node.py
```

### 2. Camera Input

Use live camera feed:
```bash
python grasp_detection_node.py --camera
```

### 3. Custom Image

Process a specific image:
```bash
python grasp_detection_node.py --image path/to/image.jpg
```

### 4. Simulate Network Conditions

Test adaptation to bandwidth changes:
```bash
# Simulate 5 Mbps bandwidth
python grasp_detection_node.py --simulate-bandwidth 5.0

# Simulate bandwidth drop during execution
python grasp_detection_node.py --simulate-bandwidth 10.0 --camera
```

## Configuration

Edit `config.yaml` to customize:
- Action space (computation distribution options)
- Optimization metric (latency, CPU, power)
- Performance thresholds
- Model parameters

## Expected Behavior

1. **High Bandwidth (>10 Mbps)**: 
   - ElasticROS tends to use cloud processing
   - Lower latency due to faster cloud inference

2. **Low Bandwidth (<5 Mbps)**:
   - ElasticROS shifts to local preprocessing
   - Reduces data transfer by sending features instead of raw images

3. **High CPU Usage**:
   - ElasticROS offloads more computation to cloud
   - Maintains responsive robot performance

## Visualization

When using camera mode:
- Green rectangles: Detected grasps
- Green circles: Grasp centers
- Green lines: Grasp orientations
- Text overlay: Confidence scores and timing information

Press:
- `q`: Quit
- `s`: Show statistics

## Performance Metrics

The example tracks:
- **Total latency**: End-to-end processing time
- **Inference time**: Cloud processing duration
- **Preprocessing level**: Where computation was performed
- **Action distribution**: How often each strategy was used

## Troubleshooting

1. **No camera detected**: 
   - Check camera connection
   - Try different camera index: `--camera-index 1`

2. **High latency**:
   - Check network connection
   - Verify AWS region is nearby
   - Consider using GPU instances

3. **Import errors**:
   - Ensure ElasticROS is installed: `pip install -e .`
   - Check Python path includes project root

## Advanced Usage

### Custom Press Node

Create your own preprocessing:

```python
class MyGraspPressNode(PressNode):
    def _process(self, data, compute_ratio):
        if compute_ratio > 0.5:
            # Do heavy preprocessing
            features = extract_deep_features(data)
            return {'features': features}
        else:
            # Minimal processing
            return {'raw_image': data}
```

### Custom Release Node

Integrate your grasp detection model:

```python
class MyGraspReleaseNode(ReleaseNode):
    def _initialize(self):
        self.model = load_my_model()
        
    def _process(self, data, compute_ratio):
        predictions = self.model.predict(data)
        return {'grasps': predictions}
```

## Benchmarking

Compare ElasticROS with fixed strategies:

```bash
cd ../benchmarks
python latency_test.py --task image --mode bandwidth
```

This will generate performance plots showing ElasticROS adaptation.
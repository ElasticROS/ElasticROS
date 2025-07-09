# ElasticROS Installation Guide

## System Requirements

### Minimum Requirements
- Ubuntu 20.04 LTS (ROS Noetic) or Ubuntu 22.04 LTS (ROS2 Humble)
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 20GB free disk space
- Internet connection (for cloud features)

### Recommended Hardware
- Multi-core CPU (4+ cores)
- NVIDIA GPU (for accelerated inference)
- Gigabit Ethernet or 802.11ac WiFi

## Installation Methods

### Method 1: Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/ElasticROS/elasticros.git
cd elasticros

# Run installation script
chmod +x scripts/install_dependencies.sh
./scripts/install_dependencies.sh

# Install Python package
pip3 install --user -e .
```

### Method 2: Manual Installation

#### 1. System Dependencies

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and development tools
sudo apt-get install -y \
    python3-pip python3-dev python3-venv \
    build-essential cmake git wget curl \
    libssl-dev libffi-dev

# Install network tools
sudo apt-get install -y \
    net-tools iperf3 \
    wondershaper
```

#### 2. Python Dependencies

```bash
# Upgrade pip
pip3 install --user --upgrade pip setuptools wheel

# Install required packages
pip3 install --user -r requirements.txt
```

#### 3. AWS CLI Installation

```bash
# Download and install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf awscliv2.zip aws/

# Configure AWS credentials
aws configure
```

#### 4. ROS Installation

##### For ROS Noetic (Ubuntu 20.04):

```bash
# Setup sources
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# Add keys
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

# Install ROS
sudo apt-get update
sudo apt-get install -y ros-noetic-ros-base python3-rosdep

# Initialize rosdep
sudo rosdep init
rosdep update

# Environment setup
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

##### For ROS2 Humble (Ubuntu 22.04):

```bash
# Add ROS2 repository
sudo apt-get install -y software-properties-common
sudo add-apt-repository universe

# Add ROS2 GPG key
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add repository to sources
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2
sudo apt-get update
sudo apt-get install -y ros-humble-ros-base python3-colcon-common-extensions

# Environment setup
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Optional Components

### GPU Support (NVIDIA)

```bash
# Check for NVIDIA GPU
nvidia-smi

# Install CUDA (if not present)
# Follow NVIDIA's official guide for your GPU

# Install Python GPU packages
pip3 install --user torch torchvision torchaudio
pip3 install --user gputil
```

### Audio Support (for speech examples)

```bash
# Install audio dependencies
sudo apt-get install -y portaudio19-dev

# Install PyAudio
pip3 install --user pyaudio
```

### Computer Vision (for image examples)

```bash
# Install OpenCV
pip3 install --user opencv-python opencv-contrib-python
```

## Verification

### 1. Check Installation

```bash
# Verify Python package
python3 -c "import elasticros_core; print(elasticros_core.__version__)"

# Verify AWS CLI
aws --version

# Verify ROS
echo $ROS_DISTRO
```

### 2. Run Tests

```bash
# Run unit tests
cd elasticros
python3 -m pytest tests/

# Test network setup
sudo ./scripts/network_setup.sh
elasticros-test-network
```

### 3. Test Examples

```bash
# Test with dummy nodes
cd examples/grasping
python3 grasp_detection_node.py --iterations 5
```

## Post-Installation Setup

### 1. Configure ElasticROS

```bash
# Create configuration directory
mkdir -p ~/.elasticros

# Copy default configs
cp config/*.yaml ~/.elasticros/

# Edit configuration
nano ~/.elasticros/default_config.yaml
```

### 2. Setup AWS Environment

```bash
# Run AWS setup wizard
python3 scripts/setup_aws.py

# Test AWS connectivity
python3 scripts/setup_aws.py --test-only
```

### 3. Network Configuration

```bash
# Configure network optimizations
sudo ./scripts/network_setup.sh

# Test bandwidth
elasticros-monitor-network
```

## Building ROS Packages

### For ROS1:

```bash
cd elasticros_ros
catkin_make
source devel/setup.bash
```

### For ROS2:

```bash
cd elasticros_ros2
colcon build
source install/setup.bash
```

## Troubleshooting

### Common Issues

1. **Import Error: elasticros_core not found**
   ```bash
   # Ensure package is installed
   pip3 install --user -e .
   
   # Check Python path
   python3 -c "import sys; print(sys.path)"
   ```

2. **AWS credentials not configured**
   ```bash
   aws configure
   # Enter your AWS Access Key ID, Secret Key, Region, and Output format
   ```

3. **ROS environment not sourced**
   ```bash
   source /opt/ros/$ROS_DISTRO/setup.bash
   # Add to ~/.bashrc for permanent setup
   ```

4. **Permission denied for network setup**
   ```bash
   # Network setup requires sudo
   sudo ./scripts/network_setup.sh
   ```

5. **CUDA/GPU not detected**
   ```bash
   # Check NVIDIA driver
   nvidia-smi
   
   # Install CUDA toolkit if missing
   # Visit: https://developer.nvidia.com/cuda-downloads
   ```

### Getting Help

- GitHub Issues: https://github.com/ElasticROS/elasticros/issues
- Documentation: https://github.com/ElasticROS/elasticros/wiki
- Email: elasticros@example.com

## Next Steps

After successful installation:

1. Read the [Quick Start Guide](quickstart.md)
2. Try the [Examples](examples/)
3. Review the [API Reference](api_reference.md)
4. Configure your robotic application

## Uninstallation

To remove ElasticROS:

```bash
# Remove Python package
pip3 uninstall elasticros

# Clean up AWS resources
python3 scripts/setup_aws.py --cleanup

# Remove configuration
rm -rf ~/.elasticros

# Remove network configurations (optional)
sudo wondershaper clear $(ip route show default | awk '/default/ {print $5}')
```
#!/bin/bash
# ElasticROS dependency installation script

set -e  # Exit on error

echo "=========================================="
echo "ElasticROS Dependency Installation"
echo "=========================================="

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    VER=$VERSION_ID
else
    echo "Cannot detect OS version"
    exit 1
fi

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo "Please do not run this script as root"
   exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Update package list
echo "Updating package list..."
sudo apt-get update

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libssl-dev \
    libffi-dev \
    net-tools \
    iperf3

# Install Python packages
echo "Installing Python dependencies..."
pip3 install --user --upgrade pip setuptools wheel

# Install network utilities
echo "Installing network utilities..."

# Install wondershaper for bandwidth control
if ! command_exists wondershaper; then
    echo "Installing wondershaper..."
    cd /tmp
    git clone https://github.com/magnific0/wondershaper.git
    cd wondershaper
    sudo make install
    cd -
    rm -rf /tmp/wondershaper
fi

# Install AWS CLI
if ! command_exists aws; then
    echo "Installing AWS CLI..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    rm -rf awscliv2.zip aws/
fi

# Check for ROS installation
echo "Checking ROS installation..."
if [ -z "$ROS_DISTRO" ]; then
    echo "WARNING: ROS not detected. ElasticROS works best with ROS installed."
    echo "Would you like to install ROS? (y/n)"
    read -r response
    
    if [ "$response" = "y" ]; then
        # Detect Ubuntu version and install appropriate ROS
        if [ "$OS" = "ubuntu" ]; then
            if [ "$VER" = "20.04" ]; then
                echo "Installing ROS Noetic..."
                sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
                curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
                sudo apt-get update
                sudo apt-get install -y ros-noetic-ros-base python3-rosdep
                
                # Initialize rosdep
                if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
                    sudo rosdep init
                fi
                rosdep update
                
                # Add to bashrc
                echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
                
            elif [ "$VER" = "22.04" ]; then
                echo "Installing ROS2 Humble..."
                sudo apt-get install -y software-properties-common
                sudo add-apt-repository universe
                
                sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
                echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
                
                sudo apt-get update
                sudo apt-get install -y ros-humble-ros-base python3-colcon-common-extensions
                
                # Add to bashrc
                echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
            else
                echo "Unsupported Ubuntu version for automatic ROS installation"
            fi
        fi
    fi
else
    echo "ROS $ROS_DISTRO detected"
fi

# Install optional dependencies
echo "Checking optional dependencies..."

# OpenCV (for image processing examples)
if ! python3 -c "import cv2" 2>/dev/null; then
    echo "Installing OpenCV..."
    pip3 install --user opencv-python opencv-contrib-python
fi

# PyAudio (for speech recognition examples)
if ! python3 -c "import pyaudio" 2>/dev/null; then
    echo "Installing PyAudio dependencies..."
    sudo apt-get install -y portaudio19-dev
    pip3 install --user pyaudio
fi

# GPU support (optional)
if command_exists nvidia-smi; then
    echo "NVIDIA GPU detected"
    echo "Installing GPU monitoring tools..."
    pip3 install --user gputil
    
    # Check for CUDA
    if [ -z "$CUDA_HOME" ]; then
        echo "WARNING: CUDA not detected. GPU acceleration may not work."
    else
        echo "CUDA detected at $CUDA_HOME"
    fi
else
    echo "No NVIDIA GPU detected - skipping GPU tools"
fi

# Create ElasticROS directories
echo "Creating ElasticROS directories..."
mkdir -p ~/.elasticros/{logs,cache,models}

# Install ElasticROS Python package
echo "Installing ElasticROS Python package..."
if [ -f setup.py ]; then
    pip3 install --user -e .
else
    echo "WARNING: setup.py not found. Please run from ElasticROS root directory."
fi

echo "=========================================="
echo "Dependency installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Configure AWS credentials: aws configure"
echo "2. Source your shell: source ~/.bashrc"
echo "3. Run tests: python3 -m pytest tests/"
echo "4. Try examples: cd examples/grasping && python3 grasp_detection_node.py"
echo ""
echo "For ROS integration:"
echo "- ROS1: cd elasticros_ros && catkin_make"
echo "- ROS2: cd elasticros_ros2 && colcon build"
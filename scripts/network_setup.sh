#!/bin/bash
# ElasticROS network configuration script

set -e

echo "=========================================="
echo "ElasticROS Network Setup"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
   echo "This script must be run as root (use sudo)"
   exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect network interface
get_default_interface() {
    # Get the interface used for default route
    ip route show default | awk '/default/ {print $5}' | head -n1
}

# Get current network configuration
INTERFACE=$(get_default_interface)
if [ -z "$INTERFACE" ]; then
    echo "Could not detect default network interface"
    echo "Available interfaces:"
    ip link show | grep -E '^[0-9]+:' | awk '{print $2}' | tr -d ':'
    echo "Please specify interface:"
    read -r INTERFACE
fi

echo "Using network interface: $INTERFACE"

# Configure network settings for ROS
echo "Configuring network for ROS..."

# Enable IP forwarding
echo "Enabling IP forwarding..."
sysctl -w net.ipv4.ip_forward=1
echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf

# Configure firewall for ROS
echo "Configuring firewall..."

# Check if ufw is installed and active
if command_exists ufw; then
    ufw_status=$(ufw status | grep -c "Status: active" || true)
    if [ "$ufw_status" -eq 1 ]; then
        echo "Configuring UFW firewall rules..."
        
        # ROS1 ports
        ufw allow 11311/tcp comment "ROS Master"
        ufw allow 11311/udp comment "ROS Master"
        
        # ROS2 DDS ports
        ufw allow 7400:7500/udp comment "ROS2 DDS Discovery"
        
        # Allow SSH
        ufw allow 22/tcp comment "SSH"
        
        # VPN ports (if using OpenVPN)
        ufw allow 1194/udp comment "OpenVPN"
        
        echo "Firewall rules configured"
    fi
fi

# Configure network optimization
echo "Applying network optimizations..."

# Increase network buffers
sysctl -w net.core.rmem_max=134217728
sysctl -w net.core.wmem_max=134217728
sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728"
sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"

# TCP optimization for low latency
sysctl -w net.ipv4.tcp_low_latency=1
sysctl -w net.ipv4.tcp_nodelay=1

# Save settings
cat >> /etc/sysctl.conf << EOF

# ElasticROS Network Optimizations
net.core.rmem_max=134217728
net.core.wmem_max=134217728
net.ipv4.tcp_rmem=4096 87380 134217728
net.ipv4.tcp_wmem=4096 65536 134217728
net.ipv4.tcp_low_latency=1
net.ipv4.tcp_nodelay=1
EOF

# Setup network monitoring
echo "Setting up network monitoring..."

# Create monitoring script
cat > /usr/local/bin/elasticros-monitor-network << 'EOF'
#!/bin/bash
# Monitor network statistics for ElasticROS

INTERFACE=${1:-$(ip route show default | awk '/default/ {print $5}' | head -n1)}

echo "Monitoring network interface: $INTERFACE"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    # Get stats
    RX_BYTES_1=$(cat /sys/class/net/$INTERFACE/statistics/rx_bytes)
    TX_BYTES_1=$(cat /sys/class/net/$INTERFACE/statistics/tx_bytes)
    
    sleep 1
    
    RX_BYTES_2=$(cat /sys/class/net/$INTERFACE/statistics/rx_bytes)
    TX_BYTES_2=$(cat /sys/class/net/$INTERFACE/statistics/tx_bytes)
    
    # Calculate rates
    RX_RATE=$((($RX_BYTES_2 - $RX_BYTES_1) * 8 / 1000000))
    TX_RATE=$((($TX_BYTES_2 - $TX_BYTES_1) * 8 / 1000000))
    
    # Get latency to Google DNS
    LATENCY=$(ping -c 1 -W 1 8.8.8.8 2>/dev/null | grep 'time=' | cut -d'=' -f4 | cut -d' ' -f1 || echo "N/A")
    
    printf "\rRX: %3d Mbps  TX: %3d Mbps  Latency: %s ms   " "$RX_RATE" "$TX_RATE" "$LATENCY"
done
EOF

chmod +x /usr/local/bin/elasticros-monitor-network

# Setup bandwidth control helper
echo "Setting up bandwidth control..."

cat > /usr/local/bin/elasticros-limit-bandwidth << 'EOF'
#!/bin/bash
# Limit bandwidth for ElasticROS testing

if [ $# -ne 3 ]; then
    echo "Usage: $0 <interface> <download_mbps> <upload_mbps>"
    echo "Example: $0 eth0 10 5"
    echo "Use 'clear' as download speed to remove limits"
    exit 1
fi

INTERFACE=$1
DOWNLOAD=$2
UPLOAD=$3

if [ "$DOWNLOAD" = "clear" ]; then
    echo "Clearing bandwidth limits on $INTERFACE"
    wondershaper clear $INTERFACE
else
    # Convert Mbps to Kbps
    DOWNLOAD_KBPS=$(($DOWNLOAD * 1000))
    UPLOAD_KBPS=$(($UPLOAD * 1000))
    
    echo "Setting bandwidth limits on $INTERFACE:"
    echo "  Download: $DOWNLOAD Mbps ($DOWNLOAD_KBPS Kbps)"
    echo "  Upload: $UPLOAD Mbps ($UPLOAD_KBPS Kbps)"
    
    wondershaper $INTERFACE $DOWNLOAD_KBPS $UPLOAD_KBPS
fi
EOF

chmod +x /usr/local/bin/elasticros-limit-bandwidth

# Create network test script
cat > /usr/local/bin/elasticros-test-network << 'EOF'
#!/bin/bash
# Test network connectivity for ElasticROS

echo "ElasticROS Network Connectivity Test"
echo "===================================="

# Test basic connectivity
echo -n "Testing internet connectivity... "
if ping -c 1 -W 2 8.8.8.8 >/dev/null 2>&1; then
    echo "OK"
else
    echo "FAILED"
    exit 1
fi

# Test DNS
echo -n "Testing DNS resolution... "
if host google.com >/dev/null 2>&1; then
    echo "OK"
else
    echo "FAILED"
fi

# Test AWS connectivity
echo -n "Testing AWS connectivity... "
if ping -c 1 -W 2 ec2.amazonaws.com >/dev/null 2>&1; then
    echo "OK"
else
    echo "WARNING: Cannot reach AWS"
fi

# Test bandwidth (if iperf3 available)
if command -v iperf3 >/dev/null 2>&1; then
    echo ""
    echo "Testing bandwidth to public iperf3 server..."
    echo "(This may take a few seconds)"
    
    # Test against public iperf3 server
    iperf3 -c iperf.he.net -t 5 -f m 2>/dev/null | grep -E "sender|receiver" || echo "Bandwidth test failed"
fi

# Show current configuration
echo ""
echo "Current Network Configuration:"
echo "=============================="
ip addr show $(ip route show default | awk '/default/ {print $5}' | head -n1) | grep -E "inet\s"

echo ""
echo "Network test complete!"
EOF

chmod +x /usr/local/bin/elasticros-test-network

# ROS-specific network setup
echo "Configuring ROS networking..."

# Create ROS network configuration script
cat > /etc/profile.d/elasticros-ros-network.sh << 'EOF'
# ElasticROS ROS Network Configuration

# Get IP address of default interface
get_ros_ip() {
    local interface=$(ip route show default | awk '/default/ {print $5}' | head -n1)
    ip addr show $interface | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n1
}

# Set ROS_IP if not already set
if [ -z "$ROS_IP" ] && [ -n "$ROS_DISTRO" ]; then
    export ROS_IP=$(get_ros_ip)
    export ROS_HOSTNAME=$ROS_IP
fi

# For ROS2, set domain ID
if [ -n "$ROS_DISTRO" ] && [ -z "$ROS_DOMAIN_ID" ]; then
    export ROS_DOMAIN_ID=0
fi
EOF

chmod +x /etc/profile.d/elasticros-ros-network.sh

echo "=========================================="
echo "Network setup complete!"
echo "=========================================="
echo ""
echo "Available commands:"
echo "  elasticros-monitor-network    - Monitor network statistics"
echo "  elasticros-limit-bandwidth    - Limit bandwidth for testing"
echo "  elasticros-test-network       - Test network connectivity"
echo ""
echo "Network optimizations applied:"
echo "  - IP forwarding enabled"
echo "  - Network buffers increased"
echo "  - TCP low latency mode enabled"
echo "  - Firewall rules configured (if applicable)"
echo ""
echo "Please log out and back in for all changes to take effect."
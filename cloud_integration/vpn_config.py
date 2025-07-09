#!/usr/bin/env python3
"""
VPN configuration for secure robot-cloud communication
Supports AWS Client VPN and OpenVPN setups
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import boto3
import time
import base64

logger = logging.getLogger(__name__)


class VPNConfig:
    """Base class for VPN configuration"""
    
    def __init__(self, config_dir: str = None):
        """Initialize VPN configuration"""
        if config_dir is None:
            config_dir = os.path.expanduser("~/.elasticros/vpn")
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Certificate paths
        self.ca_cert = self.config_dir / "ca.crt"
        self.client_cert = self.config_dir / "client.crt"
        self.client_key = self.config_dir / "client.key"
        
    def generate_certificates(self) -> bool:
        """Generate self-signed certificates for VPN"""
        try:
            # Generate CA key and certificate
            logger.info("Generating CA certificate...")
            
            # CA key
            subprocess.run([
                "openssl", "genrsa", "-out", str(self.config_dir / "ca.key"), "2048"
            ], check=True)
            
            # CA certificate
            subprocess.run([
                "openssl", "req", "-new", "-x509", "-days", "3650",
                "-key", str(self.config_dir / "ca.key"),
                "-out", str(self.ca_cert),
                "-subj", "/C=US/ST=State/L=City/O=ElasticROS/CN=ElasticROS-CA"
            ], check=True)
            
            # Generate client key and certificate
            logger.info("Generating client certificate...")
            
            # Client key
            subprocess.run([
                "openssl", "genrsa", "-out", str(self.client_key), "2048"
            ], check=True)
            
            # Client CSR
            subprocess.run([
                "openssl", "req", "-new",
                "-key", str(self.client_key),
                "-out", str(self.config_dir / "client.csr"),
                "-subj", "/C=US/ST=State/L=City/O=ElasticROS/CN=elasticros-client"
            ], check=True)
            
            # Sign client certificate
            subprocess.run([
                "openssl", "x509", "-req", "-days", "365",
                "-in", str(self.config_dir / "client.csr"),
                "-CA", str(self.ca_cert),
                "-CAkey", str(self.config_dir / "ca.key"),
                "-CAcreateserial",
                "-out", str(self.client_cert)
            ], check=True)
            
            # Set permissions
            os.chmod(self.client_key, 0o600)
            
            logger.info("Certificates generated successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate certificates: {e}")
            return False
            
    def verify_certificates(self) -> bool:
        """Verify that all required certificates exist"""
        required_files = [self.ca_cert, self.client_cert, self.client_key]
        
        for file_path in required_files:
            if not file_path.exists():
                logger.warning(f"Missing certificate file: {file_path}")
                return False
                
        logger.info("All certificates verified")
        return True


class AWSClientVPN(VPNConfig):
    """AWS Client VPN configuration"""
    
    def __init__(self, region: str = 'us-east-1', config_dir: str = None):
        super().__init__(config_dir)
        self.region = region
        self.ec2 = boto3.client('ec2', region_name=region)
        
        # VPN endpoint configuration
        self.endpoint_id = None
        self.client_config_file = self.config_dir / "elasticros-vpn.ovpn"
        
    def create_vpn_endpoint(self, 
                           client_cidr: str = "10.10.0.0/16",
                           server_cert_arn: str = None,
                           client_cert_arn: str = None) -> Optional[str]:
        """
        Create AWS Client VPN endpoint.
        
        Note: This requires certificates to be imported to AWS Certificate Manager first.
        """
        try:
            # If certificates not provided, upload them first
            if not server_cert_arn or not client_cert_arn:
                logger.info("Uploading certificates to ACM...")
                server_cert_arn = self._upload_certificate("server")
                client_cert_arn = self._upload_certificate("client")
                
            # Create VPN endpoint
            response = self.ec2.create_client_vpn_endpoint(
                ClientCidrBlock=client_cidr,
                ServerCertificateArn=server_cert_arn,
                AuthenticationOptions=[{
                    'Type': 'certificate-authentication',
                    'MutualAuthentication': {
                        'ClientRootCertificateChainArn': client_cert_arn
                    }
                }],
                ConnectionLogOptions={
                    'Enabled': False
                },
                TagSpecifications=[{
                    'ResourceType': 'client-vpn-endpoint',
                    'Tags': [
                        {'Key': 'Name', 'Value': 'ElasticROS-VPN'},
                        {'Key': 'ElasticROS', 'Value': 'true'}
                    ]
                }],
                Description='ElasticROS Client VPN for robot-cloud communication'
            )
            
            self.endpoint_id = response['ClientVpnEndpointId']
            logger.info(f"Created VPN endpoint: {self.endpoint_id}")
            
            # Wait for endpoint to be available
            waiter = self.ec2.get_waiter('client_vpn_endpoint_available')
            waiter.wait(ClientVpnEndpointIds=[self.endpoint_id])
            
            return self.endpoint_id
            
        except Exception as e:
            logger.error(f"Failed to create VPN endpoint: {e}")
            return None
            
    def associate_target_network(self, subnet_id: str) -> bool:
        """Associate VPN endpoint with target network"""
        if not self.endpoint_id:
            logger.error("No VPN endpoint ID set")
            return False
            
        try:
            response = self.ec2.associate_client_vpn_target_network(
                ClientVpnEndpointId=self.endpoint_id,
                SubnetId=subnet_id
            )
            
            association_id = response['AssociationId']
            logger.info(f"Associated VPN with subnet: {association_id}")
            
            # Wait for association
            time.sleep(10)
            
            # Authorize ingress
            self.ec2.authorize_client_vpn_ingress(
                ClientVpnEndpointId=self.endpoint_id,
                TargetNetworkCidr='0.0.0.0/0',
                AuthorizeAllGroups=True,
                Description='Allow all traffic'
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to associate target network: {e}")
            return False
            
    def download_client_configuration(self) -> bool:
        """Download VPN client configuration"""
        if not self.endpoint_id:
            logger.error("No VPN endpoint ID set")
            return False
            
        try:
            response = self.ec2.export_client_vpn_client_configuration(
                ClientVpnEndpointId=self.endpoint_id
            )
            
            config = response['ClientConfiguration']
            
            # Add certificate and key to config
            with open(self.client_cert, 'r') as f:
                cert_data = f.read()
                
            with open(self.client_key, 'r') as f:
                key_data = f.read()
                
            # Append to configuration
            config += f"\n<cert>\n{cert_data}</cert>\n"
            config += f"\n<key>\n{key_data}</key>\n"
            
            # Save configuration
            self.client_config_file.write_text(config)
            os.chmod(self.client_config_file, 0o600)
            
            logger.info(f"Client configuration saved to: {self.client_config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download client configuration: {e}")
            return False
            
    def connect(self) -> bool:
        """Connect to VPN using OpenVPN client"""
        if not self.client_config_file.exists():
            logger.error("Client configuration file not found")
            return False
            
        try:
            # Check if openvpn is installed
            subprocess.run(["which", "openvpn"], check=True, capture_output=True)
            
            # Connect using openvpn (requires sudo)
            logger.info("Connecting to VPN...")
            subprocess.Popen([
                "sudo", "openvpn", "--config", str(self.client_config_file),
                "--daemon", "elasticros-vpn"
            ])
            
            # Wait for connection
            time.sleep(5)
            
            # Check if connected
            result = subprocess.run(
                ["ip", "addr", "show", "tun0"],
                capture_output=True
            )
            
            if result.returncode == 0:
                logger.info("VPN connection established")
                return True
            else:
                logger.error("VPN connection failed")
                return False
                
        except subprocess.CalledProcessError:
            logger.error("OpenVPN not installed. Install with: sudo apt-get install openvpn")
            return False
            
    def disconnect(self):
        """Disconnect from VPN"""
        try:
            subprocess.run(["sudo", "killall", "openvpn"], check=True)
            logger.info("VPN disconnected")
        except subprocess.CalledProcessError:
            logger.warning("No active VPN connection found")
            
    def cleanup(self):
        """Clean up VPN resources"""
        if self.endpoint_id:
            try:
                # Get associations
                response = self.ec2.describe_client_vpn_target_networks(
                    ClientVpnEndpointId=self.endpoint_id
                )
                
                # Disassociate all networks
                for assoc in response['ClientVpnTargetNetworks']:
                    self.ec2.disassociate_client_vpn_target_network(
                        ClientVpnEndpointId=self.endpoint_id,
                        AssociationId=assoc['AssociationId']
                    )
                    
                # Wait a bit
                time.sleep(5)
                
                # Delete endpoint
                self.ec2.delete_client_vpn_endpoint(
                    ClientVpnEndpointId=self.endpoint_id
                )
                
                logger.info(f"Deleted VPN endpoint: {self.endpoint_id}")
                
            except Exception as e:
                logger.error(f"Failed to cleanup VPN: {e}")
                
    def _upload_certificate(self, cert_type: str) -> Optional[str]:
        """Upload certificate to AWS Certificate Manager"""
        acm = boto3.client('acm', region_name=self.region)
        
        try:
            with open(self.ca_cert, 'r') as f:
                ca_data = f.read()
                
            if cert_type == "server":
                with open(self.ca_cert, 'r') as f:
                    cert_data = f.read()
                with open(self.config_dir / "ca.key", 'r') as f:
                    key_data = f.read()
            else:
                with open(self.client_cert, 'r') as f:
                    cert_data = f.read()
                with open(self.client_key, 'r') as f:
                    key_data = f.read()
                    
            response = acm.import_certificate(
                Certificate=cert_data,
                PrivateKey=key_data,
                CertificateChain=ca_data,
                Tags=[
                    {'Key': 'Name', 'Value': f'ElasticROS-{cert_type}'},
                    {'Key': 'ElasticROS', 'Value': 'true'}
                ]
            )
            
            return response['CertificateArn']
            
        except Exception as e:
            logger.error(f"Failed to upload certificate: {e}")
            return None


class OpenVPNServer(VPNConfig):
    """OpenVPN server configuration for EC2 instances"""
    
    def __init__(self, config_dir: str = None):
        super().__init__(config_dir)
        self.server_config = self.config_dir / "server.conf"
        self.dh_params = self.config_dir / "dh2048.pem"
        
    def generate_server_config(self,
                              server_ip: str,
                              port: int = 1194,
                              protocol: str = "udp",
                              subnet: str = "10.8.0.0",
                              netmask: str = "255.255.255.0") -> bool:
        """Generate OpenVPN server configuration"""
        
        # Generate DH parameters if not exists
        if not self.dh_params.exists():
            logger.info("Generating DH parameters (this may take a while)...")
            subprocess.run([
                "openssl", "dhparam", "-out", str(self.dh_params), "2048"
            ], check=True)
            
        config = f"""# ElasticROS OpenVPN Server Configuration
port {port}
proto {protocol}
dev tun

ca {self.ca_cert}
cert {self.ca_cert}
key {self.config_dir}/ca.key
dh {self.dh_params}

server {subnet} {netmask}
ifconfig-pool-persist ipp.txt

push "redirect-gateway def1 bypass-dhcp"
push "dhcp-option DNS 8.8.8.8"
push "dhcp-option DNS 8.8.4.4"

keepalive 10 120
cipher AES-256-CBC
comp-lzo
user nobody
group nogroup
persist-key
persist-tun

status openvpn-status.log
log-append /var/log/openvpn.log
verb 3
"""
        
        self.server_config.write_text(config)
        logger.info(f"Server configuration written to: {self.server_config}")
        return True
        
    def generate_client_config(self, server_ip: str, port: int = 1194) -> str:
        """Generate client configuration"""
        
        with open(self.ca_cert, 'r') as f:
            ca_data = f.read()
            
        with open(self.client_cert, 'r') as f:
            cert_data = f.read()
            
        with open(self.client_key, 'r') as f:
            key_data = f.read()
            
        config = f"""# ElasticROS OpenVPN Client Configuration
client
dev tun
proto udp
remote {server_ip} {port}
resolv-retry infinite
nobind
user nobody
group nogroup
persist-key
persist-tun
remote-cert-tls server
cipher AES-256-CBC
comp-lzo
verb 3

<ca>
{ca_data}
</ca>

<cert>
{cert_data}
</cert>

<key>
{key_data}
</key>
"""
        
        return config
        
    def setup_server_script(self) -> str:
        """Generate script to set up OpenVPN server on EC2"""
        
        script = f"""#!/bin/bash
# ElasticROS OpenVPN Server Setup Script

# Update system
apt-get update
apt-get install -y openvpn easy-rsa

# Enable IP forwarding
echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf
sysctl -p

# Configure firewall
iptables -t nat -A POSTROUTING -s 10.8.0.0/24 -o eth0 -j MASQUERADE
iptables-save > /etc/iptables.rules

# Create OpenVPN directory
mkdir -p /etc/openvpn

# Copy configuration files
# Note: These need to be uploaded separately

# Start OpenVPN
systemctl start openvpn@server
systemctl enable openvpn@server

echo "OpenVPN server setup complete"
"""
        
        return script


def setup_robot_vpn(vpc_id: str, subnet_id: str, region: str = 'us-east-1') -> Dict[str, str]:
    """
    Complete VPN setup for robot-cloud communication.
    
    Returns:
        Dictionary with VPN configuration details
    """
    logger.info("Setting up VPN for ElasticROS...")
    
    # Create VPN configurator
    vpn = AWSClientVPN(region)
    
    # Generate certificates if needed
    if not vpn.verify_certificates():
        logger.info("Generating VPN certificates...")
        if not vpn.generate_certificates():
            raise RuntimeError("Failed to generate certificates")
            
    # Create VPN endpoint
    endpoint_id = vpn.create_vpn_endpoint()
    if not endpoint_id:
        raise RuntimeError("Failed to create VPN endpoint")
        
    # Associate with subnet
    if not vpn.associate_target_network(subnet_id):
        raise RuntimeError("Failed to associate VPN with network")
        
    # Download client configuration
    if not vpn.download_client_configuration():
        raise RuntimeError("Failed to download client configuration")
        
    return {
        'endpoint_id': endpoint_id,
        'config_file': str(vpn.client_config_file),
        'ca_cert': str(vpn.ca_cert),
        'client_cert': str(vpn.client_cert),
        'client_key': str(vpn.client_key)
    }


if __name__ == '__main__':
    # Test certificate generation
    vpn = VPNConfig()
    vpn.generate_certificates()
    print(f"Certificates generated in: {vpn.config_dir}")
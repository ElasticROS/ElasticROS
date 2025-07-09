#!/usr/bin/env python3
"""
Tests for AWS integration in ElasticROS
Note: Many tests use mocks to avoid actual AWS API calls and charges
"""

import pytest
import boto3
from moto import mock_ec2, mock_acm
from unittest.mock import Mock, patch, MagicMock
import time
import json
from pathlib import Path

from cloud_integration import AWSManager, VPCManager
from cloud_integration.vpn_config import AWSClientVPN, OpenVPNServer


class TestAWSManager:
    """Test cases for AWS Manager"""
    
    @pytest.fixture
    def aws_manager(self):
        """Create AWS Manager with mocked AWS services"""
        with mock_ec2():
            manager = AWSManager(region='us-east-1')
            yield manager
            
    @mock_ec2
    def test_find_ubuntu_ami(self, aws_manager):
        """Test finding Ubuntu AMI"""
        # Mock AMI data
        ec2 = boto3.client('ec2', region_name='us-east-1')
        
        # Create fake Ubuntu AMI
        image = ec2.register_image(
            Name='ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-test',
            Architecture='x86_64',
            RootDeviceName='/dev/sda1',
            VirtualizationType='hvm'
        )
        
        # Test finding AMI
        with patch.object(aws_manager.ec2, 'describe_images') as mock_describe:
            mock_describe.return_value = {
                'Images': [{
                    'ImageId': image['ImageId'],
                    'Name': 'ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-test',
                    'CreationDate': '2023-01-01T00:00:00.000Z',
                    'State': 'available',
                    'Architecture': 'x86_64'
                }]
            }
            
            ami_id = aws_manager.find_ubuntu_ami()
            assert ami_id == image['ImageId']
            
    @mock_ec2
    def test_setup_vpc(self, aws_manager):
        """Test VPC setup"""
        vpc_info = aws_manager.setup_vpc()
        
        assert vpc_info is not None
        assert 'vpc_id' in vpc_info
        assert 'subnet_id' in vpc_info
        assert 'igw_id' in vpc_info
        
        # Verify VPC was created
        vpcs = aws_manager.ec2.describe_vpcs(VpcIds=[vpc_info['vpc_id']])
        assert len(vpcs['Vpcs']) == 1
        assert vpcs['Vpcs'][0]['CidrBlock'] == '10.0.0.0/16'
        
    @mock_ec2
    def test_create_security_group(self, aws_manager):
        """Test security group creation"""
        # First create VPC
        vpc_response = aws_manager.ec2.create_vpc(CidrBlock='10.0.0.0/16')
        vpc_id = vpc_response['Vpc']['VpcId']
        
        # Create security group
        sg_id = aws_manager.create_security_group(vpc_id)
        
        assert sg_id is not None
        
        # Verify security group
        sgs = aws_manager.ec2.describe_security_groups(GroupIds=[sg_id])
        assert len(sgs['SecurityGroups']) == 1
        
        sg = sgs['SecurityGroups'][0]
        assert sg['GroupName'] == 'elasticros-sg'
        assert sg['VpcId'] == vpc_id
        
    @mock_ec2
    def test_launch_instance(self, aws_manager):
        """Test EC2 instance launch"""
        # Setup prerequisites
        vpc_response = aws_manager.ec2.create_vpc(CidrBlock='10.0.0.0/16')
        vpc_id = vpc_response['Vpc']['VpcId']
        
        subnet_response = aws_manager.ec2.create_subnet(
            VpcId=vpc_id,
            CidrBlock='10.0.1.0/24'
        )
        subnet_id = subnet_response['Subnet']['SubnetId']
        
        sg_id = aws_manager.create_security_group(vpc_id)
        
        # Mock AMI
        with patch.object(aws_manager, 'find_ubuntu_ami') as mock_ami:
            mock_ami.return_value = 'ami-12345678'
            
            # Mock key pair check
            with patch.object(aws_manager.ec2, 'describe_key_pairs') as mock_keys:
                mock_keys.return_value = {'KeyPairs': [{'KeyName': 'elasticros-key'}]}
                
                # Launch instance
                instance_id = aws_manager.launch_instance(
                    instance_name='test-instance',
                    instance_type='t2.micro',
                    subnet_id=subnet_id,
                    security_group_id=sg_id
                )
                
                assert instance_id is not None
                assert instance_id in aws_manager.instances.values()
                
    @mock_ec2
    def test_list_instances(self, aws_manager):
        """Test listing ElasticROS instances"""
        # Create test instances
        ec2 = boto3.resource('ec2', region_name='us-east-1')
        
        # Create instances with ElasticROS tag
        instances = ec2.create_instances(
            ImageId='ami-12345678',
            MinCount=2,
            MaxCount=2,
            InstanceType='t2.micro',
            TagSpecifications=[{
                'ResourceType': 'instance',
                'Tags': [
                    {'Key': 'ElasticROS', 'Value': 'true'},
                    {'Key': 'Name', 'Value': 'test-instance'}
                ]
            }]
        )
        
        # List instances
        instance_list = aws_manager.list_instances()
        
        assert len(instance_list) >= 2
        for inst in instance_list:
            assert 'instance_id' in inst
            assert 'state' in inst
            assert inst['name'] == 'test-instance'
            
    def test_get_instance_info(self, aws_manager):
        """Test getting instance information"""
        # Add instance to manager's tracking
        aws_manager.instances['test'] = 'i-1234567890abcdef0'
        
        # Mock EC2 resource
        mock_instance = Mock()
        mock_instance.state = {'Name': 'running'}
        mock_instance.public_ip_address = '1.2.3.4'
        mock_instance.private_ip_address = '10.0.1.5'
        mock_instance.instance_type = 't2.micro'
        mock_instance.launch_time = None
        
        with patch.object(aws_manager.ec2_resource, 'Instance') as mock_inst:
            mock_inst.return_value = mock_instance
            
            info = aws_manager.get_instance_info('test')
            
            assert info is not None
            assert info['instance_id'] == 'i-1234567890abcdef0'
            assert info['state'] == 'running'
            assert info['public_ip'] == '1.2.3.4'
            
    def test_terminate_instance(self, aws_manager):
        """Test instance termination"""
        # Add instance to tracking
        aws_manager.instances['test'] = 'i-1234567890abcdef0'
        
        # Mock termination
        with patch.object(aws_manager.ec2, 'terminate_instances') as mock_term:
            mock_term.return_value = {'TerminatingInstances': []}
            
            aws_manager.terminate_instance('test')
            
            # Verify termination was called
            mock_term.assert_called_once_with(InstanceIds=['i-1234567890abcdef0'])
            
            # Verify instance removed from tracking
            assert 'test' not in aws_manager.instances


class TestVPCManager:
    """Test cases for VPC Manager"""
    
    @pytest.fixture
    def vpc_manager(self):
        """Create VPC Manager with mocked AWS"""
        with mock_ec2():
            manager = VPCManager(region='us-east-1')
            yield manager
            
    @mock_ec2
    def test_create_elasticros_vpc(self, vpc_manager):
        """Test complete VPC creation"""
        result = vpc_manager.create_elasticros_vpc()
        
        assert 'vpc_id' in result
        assert 'public_subnet_id' in result
        assert 'private_subnet_id' in result
        assert 'internet_gateway_id' in result
        assert 'nat_gateway_id' in result
        
        # Verify VPC configuration
        vpc = vpc_manager.ec2.describe_vpcs(VpcIds=[result['vpc_id']])
        assert vpc['Vpcs'][0]['CidrBlock'] == '10.0.0.0/16'
        
    @mock_ec2
    def test_get_vpc_info(self, vpc_manager):
        """Test getting VPC information"""
        # Create VPC
        vpc_result = vpc_manager.create_elasticros_vpc()
        
        # Get info
        info = vpc_manager.get_vpc_info(vpc_result['vpc_id'])
        
        assert 'vpc' in info
        assert info['vpc']['id'] == vpc_result['vpc_id']
        assert 'subnets' in info
        assert len(info['subnets']) >= 2
        
    @mock_ec2
    def test_cleanup_vpc(self, vpc_manager):
        """Test VPC cleanup"""
        # Create VPC
        vpc_result = vpc_manager.create_elasticros_vpc()
        vpc_id = vpc_result['vpc_id']
        
        # Mock NAT gateway deletion (moto doesn't fully support)
        with patch.object(vpc_manager.ec2, 'describe_nat_gateways') as mock_nat:
            mock_nat.return_value = {'NatGateways': []}
            
            # Cleanup
            vpc_manager.cleanup_vpc(vpc_id)
            
            # Verify VPC deleted
            with pytest.raises(Exception):
                vpc_manager.ec2.describe_vpcs(VpcIds=[vpc_id])


class TestVPNConfiguration:
    """Test cases for VPN configuration"""
    
    @pytest.fixture
    def vpn_config(self, tmp_path):
        """Create VPN config with temp directory"""
        from cloud_integration.vpn_config import VPNConfig
        config = VPNConfig(config_dir=str(tmp_path))
        return config
        
    def test_generate_certificates(self, vpn_config):
        """Test certificate generation"""
        # Mock subprocess calls
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)
            
            result = vpn_config.generate_certificates()
            
            assert result is True
            
            # Verify OpenSSL commands were called
            assert mock_run.call_count >= 5  # CA key, CA cert, client key, CSR, client cert
            
    def test_verify_certificates(self, vpn_config):
        """Test certificate verification"""
        # Create dummy certificate files
        vpn_config.ca_cert.touch()
        vpn_config.client_cert.touch()
        vpn_config.client_key.touch()
        
        assert vpn_config.verify_certificates() is True
        
        # Remove one certificate
        vpn_config.client_key.unlink()
        
        assert vpn_config.verify_certificates() is False
        
    @mock_acm
    def test_aws_client_vpn(self):
        """Test AWS Client VPN setup"""
        vpn = AWSClientVPN(region='us-east-1')
        
        # Mock certificate upload
        with patch.object(vpn, '_upload_certificate') as mock_upload:
            mock_upload.return_value = 'arn:aws:acm:us-east-1:123456789012:certificate/test'
            
            # Mock VPN endpoint creation
            with patch.object(vpn.ec2, 'create_client_vpn_endpoint') as mock_create:
                mock_create.return_value = {
                    'ClientVpnEndpointId': 'cvpn-endpoint-123456'
                }
                
                endpoint_id = vpn.create_vpn_endpoint()
                
                assert endpoint_id == 'cvpn-endpoint-123456'
                assert vpn.endpoint_id == endpoint_id
                
    def test_openvpn_server_config(self, tmp_path):
        """Test OpenVPN server configuration"""
        server = OpenVPNServer(config_dir=str(tmp_path))
        
        # Mock DH parameter generation
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)
            
            # Create dummy certificate files
            server.ca_cert.touch()
            (tmp_path / "ca.key").touch()
            
            result = server.generate_server_config(
                server_ip='1.2.3.4',
                port=1194,
                subnet='10.8.0.0'
            )
            
            assert result is True
            assert server.server_config.exists()
            
            # Verify config content
            config_content = server.server_config.read_text()
            assert 'port 1194' in config_content
            assert 'server 10.8.0.0 255.255.255.0' in config_content
            
    def test_client_config_generation(self, tmp_path):
        """Test client configuration generation"""
        server = OpenVPNServer(config_dir=str(tmp_path))
        
        # Create dummy certificates
        server.ca_cert.write_text("CA CERT DATA")
        server.client_cert.write_text("CLIENT CERT DATA")
        server.client_key.write_text("CLIENT KEY DATA")
        
        config = server.generate_client_config('vpn.example.com', 1194)
        
        assert 'remote vpn.example.com 1194' in config
        assert '<ca>' in config
        assert 'CA CERT DATA' in config
        assert '<cert>' in config
        assert '<key>' in config


class TestCloudIntegration:
    """Integration tests for cloud functionality"""
    
    def test_ec2_template_loading(self):
        """Test loading EC2 instance templates"""
        template_path = Path(__file__).parent.parent / 'cloud_integration' / 'templates' / 'ec2_instance.yaml'
        
        # Check if template exists (it should in the actual project)
        if template_path.exists():
            import yaml
            
            with open(template_path, 'r') as f:
                template = yaml.safe_load(f)
                
            assert 'default' in template
            assert 'configurations' in template
            assert 'user_data_scripts' in template
            
            # Verify configurations
            assert 'image_processing' in template['configurations']
            assert 'ml_inference' in template['configurations']
            
    @patch('boto3.client')
    def test_cost_estimation(self, mock_boto_client):
        """Test cloud cost estimation"""
        from cloud_integration import AWSManager
        
        # Mock pricing API
        mock_pricing = Mock()
        mock_boto_client.return_value = mock_pricing
        
        mock_pricing.get_products.return_value = {
            'PriceList': [json.dumps({
                'product': {'attributes': {'instanceType': 't2.micro'}},
                'terms': {
                    'OnDemand': {
                        'test': {
                            'priceDimensions': {
                                'test': {'pricePerUnit': {'USD': '0.0116'}}
                            }
                        }
                    }
                }
            })]
        }
        
        # Test cost calculation
        manager = AWSManager()
        
        # Mock the release node's get_estimated_cost
        from elasticros_core import ReleaseNode
        node = ReleaseNode("test", config={'instance_type': 't2.micro'})
        
        cost = node.get_estimated_cost(1024 * 1024 * 100)  # 100MB
        
        assert cost > 0
        assert cost < 1.0  # Should be less than $1 for 100MB
        
    def test_multi_region_support(self):
        """Test multi-region cloud deployment"""
        regions = ['us-east-1', 'us-west-2', 'eu-west-1']
        
        for region in regions:
            with mock_ec2():
                manager = AWSManager(region=region)
                
                # Verify correct region
                assert manager.region == region
                assert manager.ec2._client_config.region_name == region


class TestErrorHandling:
    """Test error handling in AWS integration"""
    
    def test_invalid_credentials(self):
        """Test handling of invalid AWS credentials"""
        with patch('boto3.client') as mock_client:
            # Simulate credential error
            mock_client.side_effect = Exception("Unable to locate credentials")
            
            with pytest.raises(Exception) as exc_info:
                manager = AWSManager()
                
            assert "credentials" in str(exc_info.value).lower()
            
    def test_network_timeout(self):
        """Test handling of network timeouts"""
        manager = AWSManager()
        
        # Mock a timeout
        with patch.object(manager.ec2, 'describe_instances') as mock_describe:
            mock_describe.side_effect = TimeoutError("Network timeout")
            
            instances = manager.list_instances()
            
            # Should return empty list on error
            assert instances == []
            
    def test_instance_launch_failure(self):
        """Test handling of instance launch failures"""
        with mock_ec2():
            manager = AWSManager()
            
            # Mock launch failure
            with patch.object(manager.ec2, 'run_instances') as mock_run:
                mock_run.side_effect = Exception("InsufficientInstanceCapacity")
                
                with pytest.raises(Exception) as exc_info:
                    manager.launch_instance("test", "t2.micro", "subnet-123", "sg-123")
                    
                assert "InsufficientInstanceCapacity" in str(exc_info.value)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
#!/usr/bin/env python3
"""
AWS setup script for ElasticROS
Configures AWS environment including VPC, security groups, and key pairs
"""

import os
import sys
import boto3
import argparse
import json
import time
from pathlib import Path

# Add ElasticROS to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cloud_integration import AWSManager, VPCManager


class AWSSetup:
    """Setup AWS environment for ElasticROS"""
    
    def __init__(self, region='us-east-1'):
        self.region = region
        self.ec2 = boto3.client('ec2', region_name=region)
        
        # Managers
        self.aws_manager = AWSManager(region)
        self.vpc_manager = VPCManager(region)
        
        # Configuration
        self.config = {
            'vpc_name': 'ElasticROS-VPC',
            'key_name': 'elasticros-key',
            'security_group_name': 'elasticros-sg'
        }
        
    def check_credentials(self):
        """Check if AWS credentials are configured"""
        try:
            # Try to make a simple API call
            self.ec2.describe_regions()
            print("✓ AWS credentials configured")
            return True
        except Exception as e:
            print("✗ AWS credentials not configured")
            print(f"  Error: {e}")
            print("\n  Please run: aws configure")
            return False
            
    def create_key_pair(self):
        """Create SSH key pair for EC2 instances"""
        key_name = self.config['key_name']
        key_path = Path.home() / '.ssh' / f'{key_name}.pem'
        
        # Check if key already exists
        try:
            self.ec2.describe_key_pairs(KeyNames=[key_name])
            print(f"✓ Key pair '{key_name}' already exists")
            
            if not key_path.exists():
                print("  Warning: Private key file not found locally")
                print(f"  Expected at: {key_path}")
                
            return True
            
        except self.ec2.exceptions.ClientError:
            pass
            
        # Create new key pair
        print(f"Creating key pair '{key_name}'...")
        
        try:
            response = self.ec2.create_key_pair(
                KeyName=key_name,
                TagSpecifications=[{
                    'ResourceType': 'key-pair',
                    'Tags': [
                        {'Key': 'Name', 'Value': key_name},
                        {'Key': 'ElasticROS', 'Value': 'true'}
                    ]
                }]
            )
            
            # Save private key
            key_path.parent.mkdir(parents=True, exist_ok=True)
            key_path.write_text(response['KeyMaterial'])
            key_path.chmod(0o400)  # Set read-only for owner
            
            print(f"✓ Key pair created and saved to: {key_path}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to create key pair: {e}")
            return False
            
    def setup_vpc(self):
        """Setup VPC with all required components"""
        print("\nSetting up VPC...")
        
        try:
            # Check if VPC already exists
            vpcs = self.ec2.describe_vpcs(
                Filters=[
                    {'Name': 'tag:Name', 'Values': [self.config['vpc_name']]},
                    {'Name': 'tag:ElasticROS', 'Values': ['true']}
                ]
            )
            
            if vpcs['Vpcs']:
                vpc_id = vpcs['Vpcs'][0]['VpcId']
                print(f"✓ VPC already exists: {vpc_id}")
                
                # Get VPC info
                vpc_info = self.vpc_manager.get_vpc_info(vpc_id)
                return vpc_id, vpc_info
                
            # Create new VPC
            vpc_resources = self.vpc_manager.create_elasticros_vpc(
                name=self.config['vpc_name']
            )
            
            print(f"✓ VPC created: {vpc_resources['vpc_id']}")
            print(f"  Public subnet: {vpc_resources['public_subnet_id']}")
            print(f"  Private subnet: {vpc_resources['private_subnet_id']}")
            
            # Get full VPC info
            vpc_info = self.vpc_manager.get_vpc_info(vpc_resources['vpc_id'])
            
            return vpc_resources['vpc_id'], vpc_info
            
        except Exception as e:
            print(f"✗ Failed to setup VPC: {e}")
            return None, None
            
    def create_security_group(self, vpc_id):
        """Create security group for ElasticROS"""
        print("\nSetting up security group...")
        
        try:
            sg_id = self.aws_manager.create_security_group(vpc_id)
            print(f"✓ Security group created: {sg_id}")
            return sg_id
            
        except Exception as e:
            print(f"✗ Failed to create security group: {e}")
            return None
            
    def test_instance_launch(self, subnet_id, security_group_id):
        """Test launching an instance"""
        print("\nTesting instance launch...")
        
        try:
            # Launch test instance
            instance_id = self.aws_manager.launch_instance(
                instance_name='elasticros-test',
                instance_type='t2.micro',
                subnet_id=subnet_id,
                security_group_id=security_group_id
            )
            
            print(f"✓ Test instance launched: {instance_id}")
            
            # Get instance info
            info = self.aws_manager.get_instance_info('elasticros-test')
            print(f"  Public IP: {info['public_ip']}")
            print(f"  Private IP: {info['private_ip']}")
            
            # Ask if user wants to keep it
            response = input("\nKeep test instance running? (y/n): ")
            
            if response.lower() != 'y':
                print("Terminating test instance...")
                self.aws_manager.terminate_instance('elasticros-test')
                print("✓ Test instance terminated")
                
            return True
            
        except Exception as e:
            print(f"✗ Failed to launch test instance: {e}")
            return False
            
    def save_configuration(self, vpc_id, subnet_id, security_group_id):
        """Save configuration for ElasticROS"""
        config_dir = Path.home() / '.elasticros'
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / 'aws_config.json'
        
        config = {
            'region': self.region,
            'vpc_id': vpc_id,
            'subnet_id': subnet_id,
            'security_group_id': security_group_id,
            'key_name': self.config['key_name'],
            'setup_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        config_file.write_text(json.dumps(config, indent=2))
        print(f"\n✓ Configuration saved to: {config_file}")
        
    def print_summary(self, vpc_id, subnet_id, security_group_id):
        """Print setup summary"""
        print("\n" + "="*50)
        print("ElasticROS AWS Setup Complete!")
        print("="*50)
        
        print("\nResources created:")
        print(f"  Region: {self.region}")
        print(f"  VPC ID: {vpc_id}")
        print(f"  Subnet ID: {subnet_id}")
        print(f"  Security Group: {security_group_id}")
        print(f"  Key Pair: {self.config['key_name']}")
        
        print("\nNext steps:")
        print("1. Update your ElasticROS configuration:")
        print(f"   - VPC ID: {vpc_id}")
        print(f"   - Subnet ID: {subnet_id}")
        print(f"   - Security Group: {security_group_id}")
        print(f"   - Key Name: {self.config['key_name']}")
        
        print("\n2. Test ElasticROS:")
        print("   cd examples/grasping")
        print("   python grasp_detection_node.py")
        
        print("\n3. Monitor AWS costs:")
        print("   - Instances: https://console.aws.amazon.com/ec2")
        print("   - Billing: https://console.aws.amazon.com/billing")
        
        print("\nIMPORTANT: Remember to terminate instances when not in use!")
        
    def cleanup(self):
        """Clean up all ElasticROS resources"""
        print("\nCleaning up ElasticROS AWS resources...")
        
        # List all resources
        instances = self.aws_manager.list_instances()
        
        if instances:
            print(f"\nFound {len(instances)} running instances:")
            for inst in instances:
                print(f"  - {inst['name']} ({inst['instance_id']})")
                
            response = input("\nTerminate all instances? (yes/no): ")
            if response.lower() == 'yes':
                self.aws_manager.cleanup_all()
                print("✓ All instances terminated")
        else:
            print("No running instances found")
            
        # Check for VPCs
        vpcs = self.ec2.describe_vpcs(
            Filters=[
                {'Name': 'tag:ElasticROS', 'Values': ['true']}
            ]
        )
        
        if vpcs['Vpcs']:
            print(f"\nFound {len(vpcs['Vpcs'])} ElasticROS VPC(s)")
            response = input("Delete VPCs and all associated resources? (yes/no): ")
            
            if response.lower() == 'yes':
                for vpc in vpcs['Vpcs']:
                    print(f"Deleting VPC {vpc['VpcId']}...")
                    self.vpc_manager.cleanup_vpc(vpc['VpcId'])
                print("✓ All VPCs deleted")
                
        print("\nCleanup complete!")


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(
        description='Setup AWS environment for ElasticROS'
    )
    parser.add_argument('--region', default='us-east-1',
                       help='AWS region (default: us-east-1)')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up all ElasticROS resources')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test existing configuration')
    
    args = parser.parse_args()
    
    # Create setup instance
    setup = AWSSetup(args.region)
    
    # Check credentials first
    if not setup.check_credentials():
        return 1
        
    # Cleanup mode
    if args.cleanup:
        setup.cleanup()
        return 0
        
    # Test mode
    if args.test_only:
        # Load existing configuration
        config_file = Path.home() / '.elasticros' / 'aws_config.json'
        if not config_file.exists():
            print("No configuration found. Run setup first.")
            return 1
            
        config = json.loads(config_file.read_text())
        print(f"Testing configuration from: {config_file}")
        
        # Test launch
        setup.test_instance_launch(
            config['subnet_id'],
            config['security_group_id']
        )
        return 0
        
    # Normal setup mode
    print("Starting ElasticROS AWS setup...")
    print(f"Region: {args.region}")
    
    # Create key pair
    if not setup.create_key_pair():
        return 1
        
    # Setup VPC
    vpc_id, vpc_info = setup.setup_vpc()
    if not vpc_id:
        return 1
        
    # Get subnet ID (use public subnet)
    subnet_id = None
    for subnet in vpc_info['subnets']:
        subnet_id = subnet['id']
        break
        
    # Create security group
    sg_id = setup.create_security_group(vpc_id)
    if not sg_id:
        return 1
        
    # Test instance launch
    setup.test_instance_launch(subnet_id, sg_id)
    
    # Save configuration
    setup.save_configuration(vpc_id, subnet_id, sg_id)
    
    # Print summary
    setup.print_summary(vpc_id, subnet_id, sg_id)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
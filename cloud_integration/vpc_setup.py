#!/usr/bin/env python3
"""
VPC setup utilities for ElasticROS cloud integration
"""

import boto3
import ipaddress
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class VPCManager:
    """Manages VPC configuration for robot-cloud communication"""
    
    def __init__(self, region: str = 'us-east-1'):
        """Initialize VPC manager"""
        self.region = region
        self.ec2 = boto3.client('ec2', region_name=region)
        
        # Default network configuration
        self.vpc_cidr = '10.0.0.0/16'
        self.public_subnet_cidr = '10.0.1.0/24'
        self.private_subnet_cidr = '10.0.2.0/24'
        
    def create_elasticros_vpc(self, name: str = 'ElasticROS-VPC') -> Dict[str, str]:
        """
        Create a complete VPC setup for ElasticROS.
        
        Returns:
            Dictionary with VPC resource IDs
        """
        try:
            # Create VPC
            vpc_id = self._create_vpc(name)
            
            # Create subnets
            public_subnet_id = self._create_subnet(
                vpc_id, 
                self.public_subnet_cidr,
                f"{name}-Public-Subnet"
            )
            
            private_subnet_id = self._create_subnet(
                vpc_id,
                self.private_subnet_cidr,
                f"{name}-Private-Subnet"
            )
            
            # Create internet gateway
            igw_id = self._create_internet_gateway(vpc_id, name)
            
            # Create NAT gateway for private subnet
            nat_gateway_id = self._create_nat_gateway(public_subnet_id, name)
            
            # Configure route tables
            self._configure_public_routes(vpc_id, public_subnet_id, igw_id)
            self._configure_private_routes(vpc_id, private_subnet_id, nat_gateway_id)
            
            # Create VPN endpoint for robot connections
            vpn_endpoint_id = self._create_vpn_endpoint(vpc_id, name)
            
            result = {
                'vpc_id': vpc_id,
                'public_subnet_id': public_subnet_id,
                'private_subnet_id': private_subnet_id,
                'internet_gateway_id': igw_id,
                'nat_gateway_id': nat_gateway_id,
                'vpn_endpoint_id': vpn_endpoint_id
            }
            
            logger.info(f"Successfully created VPC setup: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error creating VPC: {e}")
            raise
            
    def _create_vpc(self, name: str) -> str:
        """Create VPC"""
        response = self.ec2.create_vpc(
            CidrBlock=self.vpc_cidr,
            TagSpecifications=[{
                'ResourceType': 'vpc',
                'Tags': [
                    {'Key': 'Name', 'Value': name},
                    {'Key': 'ElasticROS', 'Value': 'true'}
                ]
            }]
        )
        
        vpc_id = response['Vpc']['VpcId']
        
        # Enable DNS support
        self.ec2.modify_vpc_attribute(
            VpcId=vpc_id,
            EnableDnsSupport={'Value': True}
        )
        
        self.ec2.modify_vpc_attribute(
            VpcId=vpc_id,
            EnableDnsHostnames={'Value': True}
        )
        
        # Wait for VPC
        waiter = self.ec2.get_waiter('vpc_available')
        waiter.wait(VpcIds=[vpc_id])
        
        return vpc_id
        
    def _create_subnet(self, vpc_id: str, cidr: str, name: str) -> str:
        """Create subnet"""
        # Get availability zones
        azs = self.ec2.describe_availability_zones()['AvailabilityZones']
        az = azs[0]['ZoneName']  # Use first AZ
        
        response = self.ec2.create_subnet(
            VpcId=vpc_id,
            CidrBlock=cidr,
            AvailabilityZone=az,
            TagSpecifications=[{
                'ResourceType': 'subnet',
                'Tags': [
                    {'Key': 'Name', 'Value': name},
                    {'Key': 'ElasticROS', 'Value': 'true'}
                ]
            }]
        )
        
        subnet_id = response['Subnet']['SubnetId']
        
        # Enable auto-assign public IP for public subnet
        if 'Public' in name:
            self.ec2.modify_subnet_attribute(
                SubnetId=subnet_id,
                MapPublicIpOnLaunch={'Value': True}
            )
            
        return subnet_id
        
    def _create_internet_gateway(self, vpc_id: str, name: str) -> str:
        """Create and attach internet gateway"""
        # Create IGW
        response = self.ec2.create_internet_gateway(
            TagSpecifications=[{
                'ResourceType': 'internet-gateway',
                'Tags': [
                    {'Key': 'Name', 'Value': f"{name}-IGW"},
                    {'Key': 'ElasticROS', 'Value': 'true'}
                ]
            }]
        )
        
        igw_id = response['InternetGateway']['InternetGatewayId']
        
        # Attach to VPC
        self.ec2.attach_internet_gateway(
            InternetGatewayId=igw_id,
            VpcId=vpc_id
        )
        
        return igw_id
        
    def _create_nat_gateway(self, public_subnet_id: str, name: str) -> str:
        """Create NAT gateway for private subnet"""
        # Allocate Elastic IP
        eip_response = self.ec2.allocate_address(
            Domain='vpc',
            TagSpecifications=[{
                'ResourceType': 'elastic-ip',
                'Tags': [
                    {'Key': 'Name', 'Value': f"{name}-NAT-EIP"},
                    {'Key': 'ElasticROS', 'Value': 'true'}
                ]
            }]
        )
        
        allocation_id = eip_response['AllocationId']
        
        # Create NAT gateway
        response = self.ec2.create_nat_gateway(
            SubnetId=public_subnet_id,
            AllocationId=allocation_id,
            TagSpecifications=[{
                'ResourceType': 'nat-gateway',
                'Tags': [
                    {'Key': 'Name', 'Value': f"{name}-NAT"},
                    {'Key': 'ElasticROS', 'Value': 'true'}
                ]
            }]
        )
        
        nat_gateway_id = response['NatGateway']['NatGatewayId']
        
        # Wait for NAT gateway
        waiter = self.ec2.get_waiter('nat_gateway_available')
        waiter.wait(NatGatewayIds=[nat_gateway_id])
        
        return nat_gateway_id
        
    def _configure_public_routes(self, vpc_id: str, subnet_id: str, igw_id: str):
        """Configure routing for public subnet"""
        # Get main route table
        response = self.ec2.describe_route_tables(
            Filters=[
                {'Name': 'vpc-id', 'Values': [vpc_id]},
                {'Name': 'association.main', 'Values': ['true']}
            ]
        )
        
        if response['RouteTables']:
            route_table_id = response['RouteTables'][0]['RouteTableId']
        else:
            # Create new route table
            response = self.ec2.create_route_table(VpcId=vpc_id)
            route_table_id = response['RouteTable']['RouteTableId']
            
        # Add route to internet
        self.ec2.create_route(
            RouteTableId=route_table_id,
            DestinationCidrBlock='0.0.0.0/0',
            GatewayId=igw_id
        )
        
        # Associate with subnet
        self.ec2.associate_route_table(
            RouteTableId=route_table_id,
            SubnetId=subnet_id
        )
        
    def _configure_private_routes(self, vpc_id: str, subnet_id: str, nat_gateway_id: str):
        """Configure routing for private subnet"""
        # Create route table for private subnet
        response = self.ec2.create_route_table(
            VpcId=vpc_id,
            TagSpecifications=[{
                'ResourceType': 'route-table',
                'Tags': [
                    {'Key': 'Name', 'Value': 'ElasticROS-Private-RT'},
                    {'Key': 'ElasticROS', 'Value': 'true'}
                ]
            }]
        )
        
        route_table_id = response['RouteTable']['RouteTableId']
        
        # Add route through NAT gateway
        self.ec2.create_route(
            RouteTableId=route_table_id,
            DestinationCidrBlock='0.0.0.0/0',
            NatGatewayId=nat_gateway_id
        )
        
        # Associate with subnet
        self.ec2.associate_route_table(
            RouteTableId=route_table_id,
            SubnetId=subnet_id
        )
        
    def _create_vpn_endpoint(self, vpc_id: str, name: str) -> Optional[str]:
        """Create Client VPN endpoint for robot connections"""
        # This is a placeholder - actual VPN setup is complex
        # and requires certificates, client configuration, etc.
        logger.info("VPN endpoint creation is not fully implemented")
        return None
        
    def get_vpc_info(self, vpc_id: str) -> Dict:
        """Get information about VPC and its resources"""
        try:
            # Get VPC
            vpc_response = self.ec2.describe_vpcs(VpcIds=[vpc_id])
            vpc = vpc_response['Vpcs'][0]
            
            # Get subnets
            subnet_response = self.ec2.describe_subnets(
                Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}]
            )
            
            # Get route tables
            rt_response = self.ec2.describe_route_tables(
                Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}]
            )
            
            # Get internet gateways
            igw_response = self.ec2.describe_internet_gateways(
                Filters=[{'Name': 'attachment.vpc-id', 'Values': [vpc_id]}]
            )
            
            # Get NAT gateways
            nat_response = self.ec2.describe_nat_gateways(
                Filters=[
                    {'Name': 'vpc-id', 'Values': [vpc_id]},
                    {'Name': 'state', 'Values': ['available']}
                ]
            )
            
            return {
                'vpc': {
                    'id': vpc['VpcId'],
                    'cidr': vpc['CidrBlock'],
                    'state': vpc['State']
                },
                'subnets': [{
                    'id': s['SubnetId'],
                    'cidr': s['CidrBlock'],
                    'availability_zone': s['AvailabilityZone'],
                    'available_ips': s['AvailableIpAddressCount']
                } for s in subnet_response['Subnets']],
                'route_tables': len(rt_response['RouteTables']),
                'internet_gateways': len(igw_response['InternetGateways']),
                'nat_gateways': len(nat_response['NatGateways'])
            }
            
        except Exception as e:
            logger.error(f"Error getting VPC info: {e}")
            return {}
            
    def cleanup_vpc(self, vpc_id: str):
        """Clean up VPC and all associated resources"""
        logger.warning(f"Cleaning up VPC {vpc_id} and all resources...")
        
        try:
            # Delete NAT gateways first (they have EIPs)
            nat_response = self.ec2.describe_nat_gateways(
                Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}]
            )
            
            for nat in nat_response['NatGateways']:
                if nat['State'] != 'deleted':
                    self.ec2.delete_nat_gateway(NatGatewayId=nat['NatGatewayId'])
                    
            # Detach and delete internet gateways
            igw_response = self.ec2.describe_internet_gateways(
                Filters=[{'Name': 'attachment.vpc-id', 'Values': [vpc_id]}]
            )
            
            for igw in igw_response['InternetGateways']:
                self.ec2.detach_internet_gateway(
                    InternetGatewayId=igw['InternetGatewayId'],
                    VpcId=vpc_id
                )
                self.ec2.delete_internet_gateway(
                    InternetGatewayId=igw['InternetGatewayId']
                )
                
            # Delete subnets
            subnet_response = self.ec2.describe_subnets(
                Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}]
            )
            
            for subnet in subnet_response['Subnets']:
                self.ec2.delete_subnet(SubnetId=subnet['SubnetId'])
                
            # Delete route tables (except main)
            rt_response = self.ec2.describe_route_tables(
                Filters=[
                    {'Name': 'vpc-id', 'Values': [vpc_id]},
                    {'Name': 'association.main', 'Values': ['false']}
                ]
            )
            
            for rt in rt_response['RouteTables']:
                self.ec2.delete_route_table(RouteTableId=rt['RouteTableId'])
                
            # Finally delete VPC
            self.ec2.delete_vpc(VpcId=vpc_id)
            
            logger.info(f"Successfully cleaned up VPC {vpc_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up VPC: {e}")
            raise
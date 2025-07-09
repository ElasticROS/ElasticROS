#!/usr/bin/env python3
"""
ROS2 launch file for ElasticROS
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for ElasticROS"""
    
    # Launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('elasticros_ros2'),
            'config',
            'default_config.yaml'
        ]),
        description='Path to ElasticROS configuration file'
    )
    
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='elasticros',
        description='ROS2 namespace for ElasticROS nodes'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    cloud_provider_arg = DeclareLaunchArgument(
        'cloud_provider',
        default_value='aws',
        description='Cloud provider (aws, gcp, azure)'
    )
    
    instance_type_arg = DeclareLaunchArgument(
        'instance_type',
        default_value='t2.micro',
        description='Cloud instance type'
    )
    
    enable_visualization_arg = DeclareLaunchArgument(
        'enable_visualization',
        default_value='true',
        description='Enable visualization tools'
    )
    
    example_arg = DeclareLaunchArgument(
        'example',
        default_value='none',
        description='Example to run (none, image_processing, speech)'
    )
    
    # Get launch configurations
    config_file = LaunchConfiguration('config_file')
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_visualization = LaunchConfiguration('enable_visualization')
    example = LaunchConfiguration('example')
    
    # ElasticNode (main controller)
    elastic_node = Node(
        package='elasticros_ros2',
        executable='elastic_node_ros2.py',
        name='elastic_node',
        namespace=namespace,
        parameters=[{
            'config_file': config_file,
            'use_sim_time': use_sim_time,
            'optimization_metric': 'latency',
            'publish_rate': 1.0
        }],
        output='screen'
    )
    
    # Cloud Manager Node
    cloud_manager_node = Node(
        package='elasticros_ros2',
        executable='cloud_manager.py',
        name='cloud_manager',
        namespace=namespace,
        parameters=[{
            'provider': LaunchConfiguration('cloud_provider'),
            'instance_type': LaunchConfiguration('instance_type'),
            'auto_shutdown': True,
            'idle_timeout': 300
        }],
        output='screen'
    )
    
    # Performance Monitor Node
    performance_monitor_node = Node(
        package='elasticros_ros2',
        executable='performance_monitor.py',
        name='performance_monitor',
        namespace=namespace,
        parameters=[{
            'monitor_rate': 1.0,
            'metrics': ['latency', 'cpu', 'memory', 'network']
        }],
        output='screen'
    )
    
    # Visualization nodes (conditional)
    visualization_group = GroupAction(
        condition=IfCondition(enable_visualization),
        actions=[
            # RQt for monitoring
            Node(
                package='rqt_gui',
                executable='rqt_gui',
                name='elasticros_rqt',
                arguments=['--perspective-file', 
                          PathJoinSubstitution([
                              FindPackageShare('elasticros_ros2'),
                              'config',
                              'elasticros.perspective'
                          ])]
            ),
            
            # Plot juggler for real-time plots
            Node(
                package='plotjuggler',
                executable='plotjuggler',
                name='elasticros_plots',
                condition=IfCondition(LaunchConfiguration('use_plotjuggler', default='false'))
            )
        ]
    )
    
    # Example: Image Processing
    image_processing_example = GroupAction(
        condition=IfCondition(LaunchConfiguration('run_image_example', default='false')),
        actions=[
            # Camera simulator or real camera
            Node(
                package='image_publisher',
                executable='image_publisher_node',
                name='camera_simulator',
                namespace=namespace,
                parameters=[{
                    'publish_rate': 30.0,
                    'camera_info_url': ''
                }],
                remappings=[
                    ('image_raw', '/camera/image_raw'),
                    ('camera_info', '/camera/camera_info')
                ]
            ),
            
            # Image viewer
            Node(
                package='image_view',
                executable='image_view',
                name='image_viewer',
                namespace=namespace,
                remappings=[
                    ('image', 'image_processed')
                ],
                condition=IfCondition(enable_visualization)
            )
        ]
    )
    
    # Example: Speech Recognition
    speech_example = GroupAction(
        condition=IfCondition(LaunchConfiguration('run_speech_example', default='false')),
        actions=[
            Node(
                package='elasticros_ros2',
                executable='speech_example_node.py',
                name='speech_example',
                namespace=namespace,
                output='screen'
            )
        ]
    )
    
    # Logging configuration
    logging_config = Node(
        package='elasticros_ros2',
        executable='configure_logging.py',
        name='logging_config',
        parameters=[{
            'log_level': 'info',
            'log_to_file': True,
            'log_directory': '/tmp/elasticros_logs'
        }],
        output='screen'
    )
    
    # Create launch description
    ld = LaunchDescription()
    
    # Add arguments
    ld.add_action(config_file_arg)
    ld.add_action(namespace_arg)
    ld.add_action(use_sim_time_arg)
    ld.add_action(cloud_provider_arg)
    ld.add_action(instance_type_arg)
    ld.add_action(enable_visualization_arg)
    ld.add_action(example_arg)
    
    # Add nodes
    ld.add_action(elastic_node)
    ld.add_action(cloud_manager_node)
    ld.add_action(performance_monitor_node)
    ld.add_action(visualization_group)
    ld.add_action(logging_config)
    
    # Add conditional examples
    ld.add_action(image_processing_example)
    ld.add_action(speech_example)
    
    return ld
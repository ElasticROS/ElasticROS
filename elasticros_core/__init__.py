#!/usr/bin/env python3
"""
ElasticROS Core - Algorithm-level elastic computing for robotics
"""

from .elastic_node import ElasticNode
from .elastic_action import ElasticAction, Action
from .press_node import PressNode, DummyPressNode, ImageProcessingPressNode
from .release_node import ReleaseNode, DummyReleaseNode, ImageProcessingReleaseNode

__version__ = "1.0.0"
__author__ = "ElasticROS Team"

__all__ = [
    'ElasticNode',
    'ElasticAction', 
    'Action',
    'PressNode',
    'ReleaseNode',
    'DummyPressNode',
    'DummyReleaseNode',
    'ImageProcessingPressNode',
    'ImageProcessingReleaseNode',
]

# Configure logging
import logging

def setup_logging(level=logging.INFO):
    """Setup logging for ElasticROS"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Default logging setup
setup_logging()
#!/usr/bin/env python3
"""
Press Node - Base class for local computation nodes
"""

import abc
import time
import numpy as np
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PressNode(abc.ABC):
    """
    Base class for Press Nodes that execute computations locally on the robot.
    Subclass this for specific computation tasks.
    """
    
    def __init__(self, node_name: str, config: Optional[Dict] = None):
        """
        Initialize Press Node.
        
        Args:
            node_name: Unique name for this node
            config: Optional configuration dictionary
        """
        self.node_name = node_name
        self.config = config or {}
        
        # Performance tracking
        self.execution_count = 0
        self.total_execution_time = 0.0
        
        # Initialize node-specific resources
        self._initialize()
        
        logger.info(f"Initialized Press Node: {node_name}")
        
    @abc.abstractmethod
    def _initialize(self):
        """Initialize node-specific resources (models, etc)"""
        pass
        
    @abc.abstractmethod
    def _process(self, data: Any, compute_ratio: float) -> Any:
        """
        Actual processing logic to be implemented by subclasses.
        
        Args:
            data: Input data
            compute_ratio: Fraction of computation to perform (0.0-1.0)
            
        Returns:
            Processed result or intermediate state
        """
        pass
    
    def compute(self, data: Any, compute_ratio: float = 1.0) -> Any:
        """
        Execute computation with given ratio.
        
        Args:
            data: Input data
            compute_ratio: How much computation to do locally (0.0-1.0)
            
        Returns:
            Result or intermediate state for cloud processing
        """
        start_time = time.time()
        
        try:
            # Validate compute ratio
            compute_ratio = max(0.0, min(1.0, compute_ratio))
            
            # Process data
            result = self._process(data, compute_ratio)
            
            # Update stats
            self.execution_count += 1
            self.total_execution_time += (time.time() - start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Press Node {self.node_name}: {str(e)}")
            raise
    
    def prepare_for_transfer(self, data: Any, compute_ratio: float) -> Dict:
        """
        Prepare data for transfer to Release Node.
        Can include compression, serialization, etc.
        """
        # Default implementation - override for custom behavior
        return {
            'data': data,
            'compute_ratio': compute_ratio,
            'press_node': self.node_name,
            'timestamp': time.time()
        }
    
    def get_stats(self) -> Dict:
        """Get node statistics"""
        avg_time = (self.total_execution_time / self.execution_count 
                   if self.execution_count > 0 else 0.0)
        
        return {
            'node_name': self.node_name,
            'execution_count': self.execution_count,
            'average_execution_time': avg_time,
            'total_execution_time': self.total_execution_time
        }


class DummyPressNode(PressNode):
    """Example Press Node for testing"""
    
    def _initialize(self):
        """No special initialization needed"""
        pass
    
    def _process(self, data: Any, compute_ratio: float) -> Any:
        """Dummy processing - just passes through with delay"""
        # Simulate computation time based on ratio
        compute_time = 0.1 * compute_ratio  # 100ms for full computation
        time.sleep(compute_time)
        
        # Return data as-is or partially processed
        if compute_ratio >= 1.0:
            return f"Fully processed: {data}"
        else:
            return f"Partially processed ({compute_ratio:.1%}): {data}"


class ImageProcessingPressNode(PressNode):
    """Example Press Node for image processing tasks"""
    
    def _initialize(self):
        """Initialize image processing resources"""
        # Could load models, initialize GPU, etc
        self.preprocessing_steps = self.config.get('preprocessing_steps', ['resize', 'normalize'])
        
    def _process(self, data: np.ndarray, compute_ratio: float) -> Any:
        """Process image data based on compute ratio"""
        if not isinstance(data, np.ndarray):
            raise ValueError("Expected numpy array for image data")
            
        # Determine how many preprocessing steps to do locally
        num_steps = int(len(self.preprocessing_steps) * compute_ratio)
        
        result = data.copy()
        
        for i in range(num_steps):
            step = self.preprocessing_steps[i]
            
            if step == 'resize':
                # Simplified resize - in practice use cv2 or PIL
                if len(result.shape) == 3:
                    h, w = result.shape[:2]
                    # Simple downsampling
                    result = result[::2, ::2]
                    
            elif step == 'normalize':
                # Normalize to [0, 1]
                result = result.astype(np.float32)
                result = (result - result.min()) / (result.max() - result.min() + 1e-8)
                
        return {
            'processed_data': result,
            'completed_steps': num_steps,
            'remaining_steps': len(self.preprocessing_steps) - num_steps
        }
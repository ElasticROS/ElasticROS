#!/usr/bin/env python3
"""
Elastic Node - Controls computation distribution between Press and Release nodes
"""

import time
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Callable
import logging
import yaml

from .elastic_action import ElasticAction, Action
from .utils.metrics import MetricsCollector
from .utils.network_utils import NetworkMonitor

logger = logging.getLogger(__name__)


class ElasticNode:
    """
    Central controller for elastic computing.
    Manages Press/Release nodes and makes distribution decisions.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize Elastic Node with configuration"""
        # Load config
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.network_monitor = NetworkMonitor()
        
        # Initialize ElasticAction algorithm
        self.elastic_action = ElasticAction(
            action_space=self.config['action_space'],
            context_dim=self.config['context_dim'],
            gamma=self.config.get('gamma', 1.0),
            beta_multiplier=self.config.get('beta_multiplier', 1.0),
            force_sample_rate=self.config.get('force_sample_rate', 0.1)
        )
        
        # Node registry
        self.press_nodes = {}  # name -> node instance
        self.release_nodes = {}  # name -> node instance
        self.node_pairs = {}  # function_name -> (press_node, release_node)
        
        # State
        self.is_running = False
        self.current_action = None
        self.optimization_metric = self.config.get('optimization_metric', 'latency')
        
        # Performance tracking
        self.performance_history = []
        
        logger.info(f"Initialized ElasticNode with {len(self.config['action_space'])} actions")
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        if config_path is None:
            # Default config
            return {
                'action_space': [
                    {'press_ratio': 1.0, 'release_ratio': 0.0},
                    {'press_ratio': 0.5, 'release_ratio': 0.5},
                    {'press_ratio': 0.0, 'release_ratio': 1.0},
                ],
                'context_dim': 10,
                'optimization_metric': 'latency'
            }
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def register_node_pair(self, 
                          function_name: str,
                          press_node: Any,
                          release_node: Any):
        """Register a Press-Release node pair for a function"""
        self.press_nodes[function_name] = press_node
        self.release_nodes[function_name] = release_node
        self.node_pairs[function_name] = (press_node, release_node)
        
        logger.info(f"Registered node pair for function: {function_name}")
    
    def _build_context(self) -> np.ndarray:
        """Build context vector from current system state"""
        # Collect various metrics
        cpu_usage = self.metrics_collector.get_cpu_usage()
        memory_usage = self.metrics_collector.get_memory_usage()
        bandwidth = self.network_monitor.get_current_bandwidth()
        latency = self.network_monitor.get_latency()
        
        # Add temporal features
        hour = time.localtime().tm_hour / 24.0
        
        # Build context vector
        context = np.array([
            cpu_usage / 100.0,  # Normalize to [0,1]
            memory_usage / 100.0,
            min(bandwidth / 100.0, 1.0),  # Cap at 100 Mbps
            min(latency / 1000.0, 1.0),  # Cap at 1 second
            hour,
            # Add more features as needed
        ])
        
        # Pad to match context dimension
        if len(context) < self.config['context_dim']:
            context = np.pad(context, (0, self.config['context_dim'] - len(context)))
        
        return context[:self.config['context_dim']]
    
    def _get_robot_compute_costs(self) -> Dict[int, float]:
        """Estimate computation costs for each action"""
        costs = {}
        
        # Simple model - can be made more sophisticated
        base_cpu = self.metrics_collector.get_cpu_usage()
        
        for i, action_config in enumerate(self.config['action_space']):
            press_ratio = action_config['press_ratio']
            # Higher press ratio = more local computation
            costs[i] = base_cpu * press_ratio / 100.0
            
        return costs
    
    def make_decision(self, 
                     function_name: str,
                     input_data: Any = None) -> Action:
        """
        Make elastic computing decision for a function.
        
        Args:
            function_name: Name of the function to execute
            input_data: Input data (used for entropy calculation)
            
        Returns:
            Selected Action
        """
        # Build context
        context = self._build_context()
        
        # Get robot compute costs
        costs = self._get_robot_compute_costs()
        
        # Convert input data for entropy calculation if needed
        data_array = None
        if input_data is not None:
            if isinstance(input_data, np.ndarray):
                data_array = input_data
            # Add conversions for other data types as needed
        
        # Select action
        action = self.elastic_action.select_action(
            context=context,
            data=data_array,
            robot_compute_cost=costs
        )
        
        self.current_action = action
        return action
    
    def execute_with_action(self,
                           function_name: str,
                           input_data: Any,
                           action: Action) -> Tuple[Any, float]:
        """
        Execute function with given action.
        
        Returns:
            (result, execution_time)
        """
        if function_name not in self.node_pairs:
            raise ValueError(f"No node pair registered for function: {function_name}")
            
        press_node, release_node = self.node_pairs[function_name]
        
        start_time = time.time()
        
        if action.press_ratio == 1.0:
            # Full local execution
            result = press_node.compute(input_data, compute_ratio=1.0)
            
        elif action.release_ratio == 1.0:
            # Full cloud execution
            result = release_node.compute(input_data, compute_ratio=1.0)
            
        else:
            # Split execution
            # This is simplified - in practice would involve partial computation
            press_result = press_node.compute(input_data, compute_ratio=action.press_ratio)
            release_result = release_node.compute(press_result, compute_ratio=action.release_ratio)
            result = release_result
            
        execution_time = time.time() - start_time
        
        return result, execution_time
    
    def update_performance(self,
                          action: Action,
                          context: np.ndarray,
                          execution_time: float):
        """Update algorithm based on observed performance"""
        # Convert execution time to reward (negative cost)
        if self.optimization_metric == 'latency':
            reward = -execution_time
        elif self.optimization_metric == 'throughput':
            reward = 1.0 / execution_time
        else:
            reward = -execution_time  # Default to latency
            
        # Update algorithm
        self.elastic_action.update(action, context, reward)
        
        # Track performance
        self.performance_history.append({
            'timestamp': time.time(),
            'action_id': action.id,
            'execution_time': execution_time,
            'reward': reward
        })
        
    def elastic_execute(self,
                       function_name: str,
                       input_data: Any) -> Any:
        """
        Main API - execute function with elastic computing.
        
        Args:
            function_name: Registered function name
            input_data: Input data for the function
            
        Returns:
            Function result
        """
        # Make decision
        context = self._build_context()
        action = self.make_decision(function_name, input_data)
        
        # Execute
        result, execution_time = self.execute_with_action(
            function_name, input_data, action
        )
        
        # Update algorithm
        self.update_performance(action, context, execution_time)
        
        logger.debug(f"Executed {function_name} with action {action.id} "
                    f"in {execution_time:.3f}s")
        
        return result
    
    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        if not self.performance_history:
            return {}
            
        recent_history = self.performance_history[-100:]  # Last 100 executions
        
        avg_time = np.mean([h['execution_time'] for h in recent_history])
        action_counts = {}
        for h in recent_history:
            action_id = h['action_id']
            action_counts[action_id] = action_counts.get(action_id, 0) + 1
            
        return {
            'average_execution_time': avg_time,
            'action_distribution': action_counts,
            'total_executions': len(self.performance_history),
            'regret_bound': self.elastic_action.get_regret_bound()
        }
    
    def shutdown(self):
        """Clean shutdown"""
        self.is_running = False
        self.metrics_collector.stop()
        self.network_monitor.stop()
        logger.info("ElasticNode shutdown complete")
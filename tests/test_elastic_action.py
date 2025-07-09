#!/usr/bin/env python3
"""
Tests for ElasticAction algorithm
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

from elasticros_core.elastic_action import ElasticAction, Action


class TestElasticAction:
    """Test cases for ElasticAction algorithm"""
    
    @pytest.fixture
    def action_space(self):
        """Sample action space for testing"""
        return [
            {'press_ratio': 1.0, 'release_ratio': 0.0},
            {'press_ratio': 0.5, 'release_ratio': 0.5},
            {'press_ratio': 0.0, 'release_ratio': 1.0},
        ]
        
    @pytest.fixture
    def elastic_action(self, action_space):
        """Create ElasticAction instance for testing"""
        return ElasticAction(
            action_space=action_space,
            context_dim=5,
            gamma=1.0,
            beta_multiplier=1.0,
            force_sample_rate=0.1
        )
        
    def test_initialization(self, elastic_action, action_space):
        """Test proper initialization"""
        assert elastic_action.num_actions == len(action_space)
        assert elastic_action.context_dim == 5
        assert elastic_action.t == 0
        assert elastic_action.Q.shape == (5, 5)
        assert elastic_action.p.shape == (5,)
        
    def test_action_creation(self, elastic_action):
        """Test Action object creation"""
        action = elastic_action.actions[0]
        assert isinstance(action, Action)
        assert action.press_ratio == 1.0
        assert action.release_ratio == 0.0
        assert action.press_ratio + action.release_ratio == 1.0
        
    def test_select_action_first_time(self, elastic_action):
        """Test action selection on first call"""
        context = np.random.randn(5)
        
        action = elastic_action.select_action(context)
        
        assert isinstance(action, Action)
        assert elastic_action.t == 1
        
    def test_forced_sampling(self, elastic_action):
        """Test forced sampling mechanism"""
        context = np.random.randn(5)
        
        # Run until forced sampling should occur
        forced_sample_occurred = False
        
        for i in range(100):
            action = elastic_action.select_action(context)
            
            # Check if non-local action was selected
            if action.release_ratio > 0:
                forced_sample_occurred = True
                break
                
        assert forced_sample_occurred, "Forced sampling should have occurred"
        
    def test_update_mechanism(self, elastic_action):
        """Test parameter update"""
        context = np.random.randn(5)
        
        # Select cloud action
        action = elastic_action.actions[2]  # Full cloud
        
        # Update with reward
        reward = -0.5  # Negative cost
        elastic_action.update(action, context, reward)
        
        # Check that matrices were updated
        assert not np.array_equal(elastic_action.Q, elastic_action.gamma * np.eye(5))
        assert not np.array_equal(elastic_action.p, np.zeros(5))
        
    def test_entropy_weight_calculation(self, elastic_action):
        """Test entropy weight calculation"""
        # Uniform image (low entropy)
        uniform_data = np.ones((100, 100)) * 128
        low_entropy = elastic_action._compute_entropy_weight(uniform_data)
        
        # Random image (high entropy)
        random_data = np.random.randint(0, 256, (100, 100))
        high_entropy = elastic_action._compute_entropy_weight(random_data)
        
        assert 0 <= low_entropy <= 1
        assert 0 <= high_entropy <= 1
        assert high_entropy > low_entropy
        
    def test_regret_bound(self, elastic_action):
        """Test regret bound calculation"""
        # Initial regret should be 0
        assert elastic_action.get_regret_bound() == 0.0
        
        # After some steps
        elastic_action.t = 100
        regret = elastic_action.get_regret_bound()
        
        assert regret > 0
        assert regret == pytest.approx(100 ** 0.75 * np.log(100))
        
    def test_adaptive_behavior(self, elastic_action):
        """Test adaptive behavior with changing conditions"""
        # Simulate good network conditions (low latency rewards)
        good_context = np.array([0.2, 0.1, 0.9, 0.1, 0.5])  # Low CPU, high bandwidth
        
        # Run several iterations with good conditions
        for _ in range(10):
            action = elastic_action.select_action(good_context)
            if action.release_ratio > 0:
                # Cloud performed well
                elastic_action.update(action, good_context, -0.1)  # Low cost
                
        # Now simulate bad network conditions
        bad_context = np.array([0.8, 0.9, 0.1, 0.9, 0.5])  # High CPU, low bandwidth
        
        # Count local vs cloud decisions
        local_count = 0
        cloud_count = 0
        
        for _ in range(20):
            action = elastic_action.select_action(bad_context)
            
            if action.press_ratio == 1.0:
                local_count += 1
            else:
                cloud_count += 1
                # Cloud performed poorly
                elastic_action.update(action, bad_context, -2.0)  # High cost
                
        # Should prefer local computation in bad conditions
        assert local_count > cloud_count
        
    def test_reset(self, elastic_action):
        """Test reset functionality"""
        # Make some updates
        context = np.random.randn(5)
        action = elastic_action.actions[1]
        elastic_action.update(action, context, -0.5)
        elastic_action.t = 10
        
        # Reset
        elastic_action.reset()
        
        # Check reset state
        assert elastic_action.t == 0
        assert np.array_equal(elastic_action.Q, elastic_action.gamma * np.eye(5))
        assert np.array_equal(elastic_action.p, np.zeros(5))
        
    def test_concurrent_access(self, elastic_action):
        """Test thread safety of action selection"""
        import threading
        
        results = []
        context = np.random.randn(5)
        
        def select_action_thread():
            action = elastic_action.select_action(context)
            results.append(action)
            
        # Create multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=select_action_thread)
            threads.append(t)
            t.start()
            
        # Wait for completion
        for t in threads:
            t.join()
            
        # All should have completed without error
        assert len(results) == 5
        
    @pytest.mark.parametrize("context_dim", [5, 10, 20])
    def test_different_dimensions(self, action_space, context_dim):
        """Test with different context dimensions"""
        ea = ElasticAction(
            action_space=action_space,
            context_dim=context_dim
        )
        
        context = np.random.randn(context_dim)
        action = ea.select_action(context)
        
        assert isinstance(action, Action)
        assert ea.Q.shape == (context_dim, context_dim)


class TestActionSpace:
    """Test cases for action space validation"""
    
    def test_invalid_action_ratios(self):
        """Test that invalid action ratios are rejected"""
        with pytest.raises(AssertionError):
            Action(id=0, press_ratio=0.7, release_ratio=0.5)  # Sum > 1
            
    def test_empty_action_space(self):
        """Test handling of empty action space"""
        with pytest.raises((ValueError, IndexError)):
            ElasticAction(action_space=[], context_dim=5)
            
    def test_single_action(self):
        """Test with only one action available"""
        action_space = [{'press_ratio': 1.0, 'release_ratio': 0.0}]
        ea = ElasticAction(action_space=action_space, context_dim=5)
        
        context = np.random.randn(5)
        action = ea.select_action(context)
        
        # Should always select the only action
        assert action.id == 0
        assert action.press_ratio == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
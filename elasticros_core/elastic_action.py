#!/usr/bin/env python3
"""
ElasticAction online learning algorithm for adaptive resource allocation.
Based on UCB with entropy-weighted confidence bounds.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Action:
    """Represents an elastic computing action"""
    id: int
    press_ratio: float  # computation on robot
    release_ratio: float  # computation on cloud
    
    def __post_init__(self):
        # Make sure ratios sum to 1
        assert abs(self.press_ratio + self.release_ratio - 1.0) < 1e-6
        

class ElasticAction:
    """
    Online learning algorithm for elastic computing decisions.
    Uses UCB-style exploration with entropy weighting.
    """
    
    def __init__(self, 
                 action_space: List[Dict],
                 context_dim: int,
                 gamma: float = 1.0,
                 beta_multiplier: float = 1.0,
                 force_sample_rate: float = 0.1):
        """
        Initialize ElasticAction algorithm.
        
        Args:
            action_space: List of possible actions with press/release ratios
            context_dim: Dimension of context vectors
            gamma: Regularization parameter (>=1)
            beta_multiplier: Confidence bound multiplier
            force_sample_rate: Rate for forced exploration
        """
        # Convert action space to Action objects
        self.actions = [
            Action(id=i, **action) 
            for i, action in enumerate(action_space)
        ]
        self.num_actions = len(self.actions)
        self.context_dim = context_dim
        
        # Algorithm parameters
        self.gamma = max(1.0, gamma)
        self.beta_multiplier = beta_multiplier
        self.force_sample_rate = force_sample_rate
        
        # State matrices
        self.Q = self.gamma * np.eye(context_dim)  # Gram matrix
        self.p = np.zeros(context_dim)  # Reward vector
        
        # Tracking
        self.t = 0  # Time step
        self.last_forced_sample = 0
        
        logger.info(f"Initialized ElasticAction with {self.num_actions} actions")
        
    def _compute_beta(self, delta: float = 0.1) -> float:
        """Compute confidence bound parameter"""
        # Simplified version - in practice would use noise bounds
        d = self.context_dim
        t = max(1, self.t)
        
        beta = self.beta_multiplier * np.sqrt(
            d * np.log((1 + t) / delta)
        )
        return beta
    
    def _compute_entropy_weight(self, data: np.ndarray) -> float:
        """
        Compute entropy weight from data (e.g., image).
        Returns value between 0 and 1.
        """
        if data is None:
            return 0.5
            
        # Simple entropy calculation - can be customized
        if len(data.shape) == 2:  # Grayscale image
            hist, _ = np.histogram(data, bins=256, range=(0, 256))
            hist = hist / hist.sum()
            hist = hist[hist > 0]  # Remove zeros
            entropy = -np.sum(hist * np.log2(hist))
            max_entropy = np.log2(256)
            return entropy / max_entropy
        else:
            # Default for other data types
            return 0.5
    
    def _should_force_sample(self) -> bool:
        """Determine if we should force exploration"""
        if self.t == 0:
            return False
            
        # Force sample based on logarithmic schedule
        threshold = int(self.t ** self.force_sample_rate)
        if self.t >= self.last_forced_sample + threshold:
            self.last_forced_sample = self.t
            return True
        return False
    
    def select_action(self, 
                     context: np.ndarray,
                     data: Optional[np.ndarray] = None,
                     robot_compute_cost: Optional[Dict[int, float]] = None) -> Action:
        """
        Select best action based on current context.
        
        Args:
            context: Context vector describing current state
            data: Optional data (e.g., image) for entropy calculation
            robot_compute_cost: Optional dict of robot computation costs per action
            
        Returns:
            Selected Action object
        """
        self.t += 1
        
        # Get current estimate
        Q_inv = np.linalg.inv(self.Q)
        alpha_hat = Q_inv @ self.p
        
        # Compute entropy weight
        entropy_weight = 1.0 - self._compute_entropy_weight(data)
        
        # Compute confidence bounds
        beta = self._compute_beta()
        
        # Score each action
        scores = []
        for action in self.actions:
            # Expected reward
            expected = context.T @ alpha_hat
            
            # Confidence bound
            confidence = beta * np.sqrt(
                entropy_weight * context.T @ Q_inv @ context
            )
            
            # Robot compute cost (if provided)
            cost = robot_compute_cost.get(action.id, 0.0) if robot_compute_cost else 0.0
            
            # UCB score (minimize)
            score = cost + expected - confidence
            scores.append(score)
        
        # Force exploration if needed
        if self._should_force_sample():
            # Sample from non-pure-robot actions
            valid_actions = [i for i, a in enumerate(self.actions) if a.release_ratio > 0]
            if valid_actions:
                best_action_idx = np.random.choice(valid_actions)
            else:
                best_action_idx = np.argmin(scores)
        else:
            best_action_idx = np.argmin(scores)
        
        selected = self.actions[best_action_idx]
        logger.debug(f"Step {self.t}: Selected action {selected.id} "
                    f"(press={selected.press_ratio:.2f}, release={selected.release_ratio:.2f})")
        
        return selected
    
    def update(self, 
               action: Action,
               context: np.ndarray,
               reward: float):
        """
        Update algorithm parameters based on observed reward.
        
        Args:
            action: Action that was taken
            context: Context when action was taken
            reward: Observed reward (negative for costs)
        """
        # Only update if we actually observed cloud performance
        if action.release_ratio > 0:
            self.Q += np.outer(context, context)
            self.p += context * reward
            
            logger.debug(f"Updated parameters with reward={reward:.4f}")
    
    def get_regret_bound(self) -> float:
        """Get theoretical regret bound at current time step"""
        if self.t == 0:
            return 0.0
            
        # O(T^0.75 * log(T)) bound from paper
        return self.t ** 0.75 * np.log(self.t)
    
    def reset(self):
        """Reset algorithm state"""
        self.Q = self.gamma * np.eye(self.context_dim)
        self.p = np.zeros(self.context_dim)
        self.t = 0
        self.last_forced_sample = 0
        logger.info("Reset ElasticAction algorithm state")
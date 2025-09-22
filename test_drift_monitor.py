#!/usr/bin/env python3
"""
Test script for adaptive DPO with topic drift monitoring.
Run this to validate the new drift monitoring mechanism.
"""

import os
import sys
import torch
import numpy as np
from utils.topic_drift_monitor import TopicDriftMonitor

def test_drift_monitoring():
    """Test topic drift monitoring functionality."""
    
    print("🧪 Testing Topic Drift Monitor...")
    
    # Create sample vocabulary
    vocab = [f"word_{i}" for i in range(1000)]
    
    # Initialize monitor
    monitor = TopicDriftMonitor(
        vocab=vocab,
        num_topics=10,
        top_k=25,
        jaccard_threshold=0.6,
        check_interval=5
    )
    
    # Test 1: Initial beta (no drift)
    print("\n📊 Test 1: Initial beta matrix")
    beta_init = torch.randn(10, 1000)
    beta_init = torch.softmax(beta_init, dim=1)
    
    # Initialize reference
    monitor.initialize_reference(beta_init)
    
    # Check drift (should be no drift)
    drift_result = monitor.check_drift(beta_init, epoch=5)
    print(f"   Drift detected: {drift_result['drift_detected']}")
    print(f"   Mean Jaccard: {drift_result.get('mean_jaccard', 'N/A'):.3f}")
    
    # Test 2: Slightly drifted beta
    print("\n📊 Test 2: Slightly drifted topics")
    beta_drift_slight = beta_init.clone()
    # Add small noise to simulate slight drift
    noise = torch.randn_like(beta_drift_slight) * 0.01
    beta_drift_slight = torch.softmax(beta_drift_slight + noise, dim=1)
    
    drift_result = monitor.check_drift(beta_drift_slight, epoch=10)
    print(f"   Drift detected: {drift_result['drift_detected']}")
    print(f"   Mean Jaccard: {drift_result.get('mean_jaccard', 'N/A'):.3f}")
    
    # Test 3: Heavily drifted beta (should trigger refresh)
    print("\n📊 Test 3: Heavily drifted topics")
    beta_drift_heavy = torch.randn(10, 1000)  # Completely new distribution
    beta_drift_heavy = torch.softmax(beta_drift_heavy, dim=1)
    
    drift_result = monitor.check_drift(beta_drift_heavy, epoch=15)
    print(f"   Drift detected: {drift_result['drift_detected']}")
    print(f"   Should refresh: {drift_result['should_refresh']}")
    print(f"   Mean Jaccard: {drift_result.get('mean_jaccard', 'N/A'):.3f}")
    print(f"   Drifted topics: {drift_result.get('num_drifted', 0)}/10")
    
    # Test 4: Jaccard computation
    print("\n📊 Test 4: Manual Jaccard computation")
    set_a = {1, 2, 3, 4, 5}
    set_b = {3, 4, 5, 6, 7}  
    jaccard = monitor.compute_jaccard_similarity(set_a, set_b)
    expected = 3 / 7  # intersection=3, union=7
    print(f"   Set A: {set_a}")
    print(f"   Set B: {set_b}")
    print(f"   Jaccard: {jaccard:.3f} (expected: {expected:.3f})")
    print(f"   ✅ Correct: {abs(jaccard - expected) < 1e-6}")
    
    print("\n✅ All drift monitoring tests completed!")

if __name__ == "__main__":
    test_drift_monitoring()

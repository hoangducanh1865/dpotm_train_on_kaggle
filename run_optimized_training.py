#!/usr/bin/env python3
"""
Optimized training script with enhanced hyperparameters for improved TC_15 scores
"""

import subprocess
import sys
import os

def run_training_with_optimized_params():
    """
    Run ECRTM training with optimized hyperparameters for better TC_15 scores
    """
    
    # Base command
    base_cmd = [
        sys.executable, "main.py",
        "--model", "ECRTM",
        "--dataset", "BBC_new",  # Change this to your dataset
        "--num_topics", "50",
        "--device", "cuda",
        "--use_pretrainWE"
    ]
    
    # Optimized hyperparameters based on analysis
    optimized_params = [
        # Core ECRTM parameters (tuned ranges)
        "--weight_ECR", "175",  # Increased from 100 to 175 (middle of 150-200 range)
        "--weight_InfoNCE", "125",  # Increased from 50 to 125 (middle of 100-150 range)
        "--beta_temp", "0.12",  # Reduced from 0.2 to 0.12 (middle of 0.1-0.15 range)
        
        # DPO fine-tuning parameters
        "--lambda_dpo", "0.6",  # Slightly increased for stronger preference learning
        "--lambda_reg", "0.008",  # Increased regularization
        "--lambda_diversity", "0.15",  # Topic diversity regularization
        "--use_ipo",  # Use IPO for more stable training
        "--label_smoothing", "0.05",  # Small amount of label smoothing
        
        # Training parameters
        "--dropout", "0.1",  # Reduced dropout for better learning
        "--hidden_dim_1", "512",  # Increased capacity
        "--hidden_dim_2", "512",
    ]
    
    # Combine all parameters
    full_cmd = base_cmd + optimized_params
    
    print("Running optimized ECRTM training with parameters:")
    print(" ".join(full_cmd))
    print("\n" + "="*80)
    
    # Run the training
    try:
        result = subprocess.run(full_cmd, check=True, capture_output=False)
        print("\n" + "="*80)
        print("Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code: {e.returncode}")
        return False

def run_hyperparameter_sweep():
    """
    Run multiple training runs with different hyperparameter combinations
    """
    
    # Different parameter combinations to try
    param_combinations = [
        {
            "name": "conservative",
            "weight_ECR": "150",
            "weight_InfoNCE": "100", 
            "beta_temp": "0.15",
            "lambda_dpo": "0.5",
            "lambda_diversity": "0.1"
        },
        {
            "name": "aggressive", 
            "weight_ECR": "200",
            "weight_InfoNCE": "150",
            "beta_temp": "0.10",
            "lambda_dpo": "0.7",
            "lambda_diversity": "0.2"
        },
        {
            "name": "balanced",
            "weight_ECR": "175",
            "weight_InfoNCE": "125",
            "beta_temp": "0.12", 
            "lambda_dpo": "0.6",
            "lambda_diversity": "0.15"
        }
    ]
    
    results = {}
    
    for i, params in enumerate(param_combinations):
        print(f"\n{'='*60}")
        print(f"Running configuration {i+1}/3: {params['name']}")
        print(f"{'='*60}")
        
        base_cmd = [
            sys.executable, "main.py",
            "--model", "ECRTM",
            "--dataset", "20NG",
            "--num_topics", "50", 
            "--device", "cuda",
            "--use_pretrainWE"
        ]
        
        param_cmd = [
            "--weight_ECR", params["weight_ECR"],
            "--weight_InfoNCE", params["weight_InfoNCE"],
            "--beta_temp", params["beta_temp"],
            "--lambda_dpo", params["lambda_dpo"],
            "--lambda_reg", "0.008",
            "--lambda_diversity", params["lambda_diversity"],
            "--use_ipo",
            "--label_smoothing", "0.05"
        ]
        
        full_cmd = base_cmd + param_cmd
        
        try:
            result = subprocess.run(full_cmd, check=True, capture_output=True, text=True)
            results[params['name']] = "SUCCESS"
            print(f"Configuration '{params['name']}' completed successfully")
        except subprocess.CalledProcessError as e:
            results[params['name']] = f"FAILED (code: {e.returncode})"
            print(f"Configuration '{params['name']}' failed")
    
    print(f"\n{'='*60}")
    print("HYPERPARAMETER SWEEP RESULTS:")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"{name:15}: {status}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run optimized ECRTM training")
    parser.add_argument("--sweep", action="store_true", 
                       help="Run hyperparameter sweep with multiple configurations")
    parser.add_argument("--dataset", type=str, default="20NG",
                       help="Dataset to use (20NG, BBC_new, WOS_vocab_5k, etc.)")
    
    args = parser.parse_args()
    
    if args.sweep:
        print("Starting hyperparameter sweep...")
        run_hyperparameter_sweep()
    else:
        print("Starting single optimized training run...")
        success = run_training_with_optimized_params()
        sys.exit(0 if success else 1)

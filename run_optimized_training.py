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
        "--use_pretrainWE",
        "--finetune_epochs", "100"  # Only fine-tune 100 epochs after loading checkpoint
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
        print("🚀 Starting training...")
        result = subprocess.run(full_cmd, check=True, capture_output=True, text=True)
        
        # Extract and display key results
        output_lines = result.stdout.split('\n')
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Extract key metrics from output
        metrics = {}
        for line in output_lines:
            if "TC_15:" in line:
                metrics['TC_15'] = line.strip()
            elif "NMI:" in line:
                metrics['NMI'] = line.strip()
            elif "Purity:" in line:
                metrics['Purity'] = line.strip() 
            elif "Accuracy:" in line:
                metrics['Accuracy'] = line.strip()
            elif "Macro-f1" in line:
                metrics['Macro-F1'] = line.strip()
        
        # Display extracted metrics
        if metrics:
            print("\n📊 KEY RESULTS:")
            print("-" * 40)
            for metric, value in metrics.items():
                print(f"  {value}")
        
        # Show last few lines of output for additional context
        print("\n📝 TRAINING LOG (last 10 lines):")
        print("-" * 40)
        for line in output_lines[-10:]:
            if line.strip():
                print(f"  {line}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code: {e.returncode}")
        if e.stderr:
            print("Error details:")
            print(e.stderr)
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
            "--dataset", "BBC_new",
            "--num_topics", "50", 
            "--device", "cuda",
            "--use_pretrainWE",
            "--finetune_epochs", "100"  # Only fine-tune 100 epochs
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
            print(f"🚀 Running {params['name']} configuration...")
            result = subprocess.run(full_cmd, check=True, capture_output=True, text=True)
            
            # Extract TC_15 score for comparison
            tc_15_score = None
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if "TC_15:" in line:
                    try:
                        tc_15_score = float(line.split(":")[-1].strip())
                    except:
                        tc_15_score = line.split(":")[-1].strip()
                    break
            
            results[params['name']] = {
                'status': 'SUCCESS',
                'TC_15': tc_15_score,
                'output': result.stdout
            }
            print(f"✅ Configuration '{params['name']}' completed - TC_15: {tc_15_score}")
            
        except subprocess.CalledProcessError as e:
            results[params['name']] = {
                'status': f'FAILED (code: {e.returncode})',
                'TC_15': None,
                'error': e.stderr
            }
            print(f"❌ Configuration '{params['name']}' failed")
    
    print(f"\n{'='*60}")
    print("🏆 HYPERPARAMETER SWEEP RESULTS:")
    print(f"{'='*60}")
    
    # Sort results by TC_15 score if available
    successful_results = [(name, data) for name, data in results.items() 
                         if data['status'] == 'SUCCESS' and data['TC_15'] is not None]
    
    if successful_results:
        # Sort by TC_15 score (descending)
        successful_results.sort(key=lambda x: float(x[1]['TC_15']) if isinstance(x[1]['TC_15'], (int, float)) else 0, reverse=True)
        
        print("\n📊 RANKED BY TC_15 SCORE:")
        print("-" * 40)
        for i, (name, data) in enumerate(successful_results, 1):
            tc_15 = data['TC_15']
            print(f"{i}. {name:12}: TC_15 = {tc_15}")
            
        # Show best configuration details
        if successful_results:
            best_name, best_data = successful_results[0]
            print(f"\n🥇 BEST CONFIGURATION: {best_name}")
            print("-" * 40)
            
            # Extract all metrics from best result
            best_output = best_data['output'].split('\n')
            metrics = {}
            for line in best_output:
                for metric_name in ["TC_15:", "NMI:", "Purity:", "Accuracy:", "Macro-f1"]:
                    if metric_name in line:
                        metrics[metric_name.rstrip(':')] = line.strip()
            
            for metric, value in metrics.items():
                print(f"  {value}")
    
    # Show all results summary
    print(f"\n📋 COMPLETE SUMMARY:")
    print("-" * 40)
    for name, data in results.items():
        status = data['status']
        tc_15 = data.get('TC_15', 'N/A')
        print(f"{name:15}: {status} | TC_15: {tc_15}")
    
    return results

if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Run optimized ECRTM training")
    parser.add_argument("--sweep", action="store_true", 
                       help="Run hyperparameter sweep with multiple configurations")
    parser.add_argument("--dataset", type=str, default="BBC_new",
                       help="Dataset to use (20NG, BBC_new, WOS_vocab_5k, etc.)")
    parser.add_argument("--save_results", action="store_true",
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    if args.sweep:
        print("Starting hyperparameter sweep...")
        results = run_hyperparameter_sweep()
        
        # Save results if requested
        if args.save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"hyperparameter_sweep_results_{timestamp}.json"
            
            # Convert results to serializable format
            save_data = {
                'timestamp': timestamp,
                'dataset': args.dataset,
                'results': {}
            }
            
            for name, data in results.items():
                save_data['results'][name] = {
                    'status': data['status'],
                    'TC_15': data.get('TC_15'),
                    'error': data.get('error', None)
                }
            
            with open(results_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"\n💾 Results saved to: {results_file}")
            
    else:
        print("Starting single optimized training run...")
        success = run_training_with_optimized_params()
        sys.exit(0 if success else 1)

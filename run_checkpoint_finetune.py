#!/usr/bin/env python3
"""
Script to run DPO fine-tuning from checkpoint_epoch_500.pt
Only performs 100 epochs of fine-tuning for improved TC_15 scores
"""

import subprocess
import sys
import os

def run_checkpoint_finetuning():
    """
    Run DPO fine-tuning from existing checkpoint with optimized parameters
    """
    
    print("🔄 Starting DPO fine-tuning from checkpoint_epoch_500.pt...")
    print("📊 Target: Improve TC_15 from ~0.43 to 0.45+")
    
    # Fine-tuning command from checkpoint
    finetune_cmd = [
        sys.executable, "main.py",
        "--model", "ECRTM",
        "--dataset", "BBC_new",
        "--num_topics", "50",
        "--device", "cuda",
        "--use_pretrainWE",
        
        # Fine-tuning specific settings
        "--finetune_epochs", "100",  # Only 100 epochs fine-tuning
        
        # Optimized hyperparameters for better TC_15
        "--weight_ECR", "175",
        "--weight_InfoNCE", "125", 
        "--beta_temp", "0.12",
        
        # DPO fine-tuning parameters
        "--lambda_dpo", "0.6",
        "--lambda_reg", "0.008", 
        "--lambda_diversity", "0.15",
        "--use_ipo",
        "--label_smoothing", "0.05",
        
        # Enhanced model capacity
        "--dropout", "0.1",
        "--hidden_dim_1", "512",
        "--hidden_dim_2", "512"
    ]
    
    print("Command to execute:")
    print(" ".join(finetune_cmd))
    print("\n" + "="*80)
    print("🚀 STARTING DPO FINE-TUNING FROM CHECKPOINT...")
    print("="*80)
    
    try:
        # Run with real-time output
        result = subprocess.run(finetune_cmd, check=True, capture_output=False)
        
        print("\n" + "="*80)
        print("✅ DPO FINE-TUNING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("📈 Check the output above for improved TC_15 scores")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Fine-tuning failed with error code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️ Fine-tuning interrupted by user")
        return False

def run_quick_test():
    """
    Quick test with just 5 epochs to verify everything works
    """
    
    print("🧪 Running quick test (5 epochs) to verify DPO fine-tuning...")
    
    test_cmd = [
        sys.executable, "main.py",
        "--model", "ECRTM", 
        "--dataset", "BBC_new",
        "--num_topics", "20",  # Smaller for testing
        "--device", "cuda",
        "--use_pretrainWE",
        "--finetune_epochs", "5",  # Quick test
        
        # Basic DPO parameters
        "--lambda_dpo", "0.5",
        "--lambda_reg", "0.005",
        "--lambda_diversity", "0.1"
    ]
    
    try:
        result = subprocess.run(test_cmd, check=True, capture_output=True, text=True)
        print("✅ Quick test PASSED - DPO fine-tuning is working!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Quick test FAILED: {e.returncode}")
        if e.stderr:
            print("Error:", e.stderr)
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run DPO fine-tuning from checkpoint")
    parser.add_argument("--test", action="store_true",
                       help="Run quick test (5 epochs) instead of full fine-tuning")
    parser.add_argument("--dataset", type=str, default="BBC_new",
                       help="Dataset to use")
    
    args = parser.parse_args()
    
    if args.test:
        success = run_quick_test()
    else:
        print("💡 IMPORTANT: Make sure checkpoint_epoch_500.pt exists in results directory")
        print("📁 Expected path: results/ECRTM/{dataset}/{timestamp}/checkpoint_epoch_500.pt\n")
        
        confirm = input("Continue with 100-epoch DPO fine-tuning? [y/N]: ")
        if confirm.lower() in ['y', 'yes']:
            success = run_checkpoint_finetuning()
        else:
            print("❌ Fine-tuning cancelled")
            success = False
    
    sys.exit(0 if success else 1)

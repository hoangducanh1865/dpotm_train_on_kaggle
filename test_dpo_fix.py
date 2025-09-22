#!/usr/bin/env python3
"""
Quick test script to verify the DPO fine-tuning fix
"""

import subprocess
import sys
import os

def test_dpo_fix():
    """
    Test the DPO fine-tuning fix with minimal configuration
    """
    
    print("🧪 Testing DPO fine-tuning fix...")
    
    # Simple test command with minimal epochs
    test_cmd = [
        sys.executable, "main.py",
        "--model", "ECRTM",
        "--dataset", "BBC_new",
        "--num_topics", "20",  # Reduced for faster testing
        "--device", "cuda",
        "--use_pretrainWE",
        
        # DPO parameters 
        "--lambda_dpo", "0.5",
        "--lambda_reg", "0.005",
        "--lambda_diversity", "0.1",
        "--use_ipo",
        
        # Minimal training for testing (since checkpoint already loaded)
        "--finetune_epochs", "5"  # Very short test for fix verification
    ]
    
    print("Running test command:")
    print(" ".join(test_cmd))
    print("\n" + "="*60)
    
    try:
        result = subprocess.run(test_cmd, check=True, capture_output=True, text=True)
        
        print("✅ DPO fine-tuning fix test PASSED!")
        print("="*60)
        
        # Show relevant output
        output_lines = result.stdout.split('\n')
        
        # Look for DPO-related output
        dpo_lines = []
        for line in output_lines:
            if any(keyword in line.lower() for keyword in ['dpo', 'preference', 'fine', 'epoch']):
                dpo_lines.append(line)
        
        if dpo_lines:
            print("\n📊 DPO Training Output:")
            print("-" * 40)
            for line in dpo_lines[-10:]:  # Last 10 relevant lines
                if line.strip():
                    print(f"  {line}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Test FAILED with error code: {e.returncode}")
        print("Error details:")
        if e.stderr:
            print(e.stderr)
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        return False

if __name__ == "__main__":
    success = test_dpo_fix()
    sys.exit(0 if success else 1)

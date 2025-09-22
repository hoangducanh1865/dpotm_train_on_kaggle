"""
🚀 TEST SCRIPT: Enhanced DPO Implementation
Test the improved ECRTM with advanced LLM DPO techniques
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from models.ECRTM.ECRTM import ECRTM
from datasethandler.basic_dataset_handler import BasicDatasetHandler
from utils.config import Config

def test_enhanced_dpo():
    print("🧪 Testing Enhanced DPO Implementation...")
    
    # Load config with enhanced DPO parameters
    config = Config()
    print(f"📋 Enhanced DPO Config:")
    print(f"   - lambda_dpo: {config.lambda_dpo}")
    print(f"   - lambda_reg: {config.lambda_reg}")
    print(f"   - use_ipo: {config.use_ipo}")
    print(f"   - label_smoothing: {config.label_smoothing}")
    
    # Initialize dataset
    print("📚 Loading dataset...")
    dataset_name = "20NG"
    dataset_handler = BasicDatasetHandler(dataset_name, device='cuda')
    
    # Initialize enhanced ECRTM model
    print("🤖 Initializing Enhanced ECRTM...")
    model = ECRTM(
        vocab_size=len(dataset_handler.vocab),
        num_topics=config.num_topics,
        num_hiddens=config.num_hiddens,
        num_clusters=config.num_clusters,
        dropout=config.dropout,
        learn_priors=config.learn_priors,
        pretrained_WE=dataset_handler.wordembeddings,
        device='cuda',
        # Enhanced DPO parameters
        lambda_dpo=config.lambda_dpo,
        lambda_reg=config.lambda_reg,
        use_ipo=config.use_ipo,
        label_smoothing=config.label_smoothing
    ).cuda()
    
    print(f"✅ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    print("🔄 Testing forward pass...")
    batch_data = next(iter(dataset_handler.train_dataloader))
    
    with torch.no_grad():
        result = model(batch_data)
        
    print("📊 Forward pass results:")
    for key, value in result.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, torch.Tensor):
                    print(f"      {sub_key}: {sub_val.item():.4f}")
                else:
                    print(f"      {sub_key}: {sub_val:.4f}")
        elif isinstance(value, torch.Tensor):
            print(f"   {key}: {value.item():.4f}")
        else:
            print(f"   {key}: {value:.4f}")
    
    print("\n🎯 Enhanced DPO Implementation Test Completed!")
    print("Ready for training to achieve TC_15 → 0.50+ target!")

if __name__ == "__main__":
    test_enhanced_dpo()

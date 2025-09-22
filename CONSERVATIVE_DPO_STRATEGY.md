# 🎯 FINAL STRATEGY: Conservative DPO for TC_15 Preservation

## 📊 **NEW GOAL**: Protect TC_15 base score + minimal improvement

## ⚠️ **Critical Discovery**: 
- **Before DPO**: TC_15 = 0.43249 ✅
- **After DPO**: TC_15 = 0.42467 ❌ (DEGRADED!)
- **Problem**: DPO fine-tuning is DESTROYING topic coherence

## 🚨 **ROOT CAUSE ANALYSIS**:
1. **DPO conflict**: Preference optimization competing with topic coherence
2. **Overfitting**: Too many fine-tune epochs damaging learned representations
3. **Learning rate**: Too aggressive fine-tuning disrupting base model

## 🔧 **CONSERVATIVE STRATEGY**:

### **1. Enhance Base Model (500 epochs)**
Focus on improving TC_15 BEFORE DPO:

| Parameter | Old | **NEW** | Impact |
|-----------|-----|---------|--------|
| `TW_alpha` | 2.5 | **3.5** | 🎯 STRONG topic-word coherence |
| `DT_alpha` | 3.5 | **2.5** | Less diversity penalty |
| `weight_InfoNCE` | 60.0 | **80.0** | Better contrastive learning |
| `beta_temp` | 0.2 | **0.15** | Sharper word distributions |
| `weight_ECR` | 120.0 | **150.0** | Better clustering |

**Target**: Get base TC_15 from 0.43249 → **0.45-0.46**

### **2. MINIMAL DPO (50 epochs only)**
Extremely gentle fine-tuning to preserve base quality:

| Parameter | Old | **NEW** | Impact |
|-----------|-----|---------|--------|
| `lambda_dpo` | 1.0 | **0.3** | 🚨 MUCH weaker DPO |
| `lambda_reg` | 0.015 | **0.005** | Minimal regularization |
| `use_ipo` | True | **False** | Disable IPO (may conflict) |
| `label_smoothing` | 0.1 | **0.05** | Minimal smoothing |
| `finetune_epochs` | 100 | **50** | 🚨 HALF the epochs |
| `finetune_lr` | 0.001 | **0.0005** | 🚨 HALF the learning rate |

**Target**: Preserve TC_15 ≥ 0.45, ideally +0.01-0.02

## 🎯 **Expected Results**:

```
Base Training (500 epochs):
├── TC_15: 0.43249 → 0.45-0.46 (+4-6%)
├── Better TW_alpha = stronger coherence
└── Enhanced clustering & contrastive learning

Conservative DPO (50 epochs):
├── TC_15: 0.45-0.46 → 0.46-0.47 (+0.01-0.02)
├── Minimal DPO interference
└── Preserve base model quality
```

## 🏃‍♂️ **Execution**:

```bash
# Run with conservative parameters
python main.py

# Expected timeline:
# - 500 epochs base: Focus on TC_15 improvement
# - 50 epochs DPO: Gentle preference optimization
# - Total target: TC_15 ≥ 0.46-0.47
```

## 🔑 **Key Philosophy**:
**"Better to preserve good base model than risk destroying it with aggressive DPO"**

- Priority 1: Get strong base TC_15 (0.45+)
- Priority 2: Minimal DPO interference
- Priority 3: Preserve coherence above all

**Success metric**: TC_15 after DPO ≥ TC_15 before DPO!

# Enhanced DPO Strategy for TC_15 Improvement

## Analysis of Previous Results

TC_15: 0.44019 → 0.44049 (+0.0003) - **Cải thiện quá nhỏ**

**Root Cause**: DPO signal còn quá yếu, chưa đủ để drive meaningful improvements

## Enhanced Strategy

### 1. Improved Preference Pairing Quality
**Before**: 10 pairs/topic với random pairing
**Now**: 20 pairs/topic với strategic pairing:
- **5 pairs** Best-vs-Worst cho strong signal
- **15 pairs** balanced random cho diversity

```python
# Strategy 1: Best vs Worst cho strong signal
for i in range(min(5, min_pairs)):
    w_plus_idx = w_plus_indices[i]      # Top words
    w_minus_idx = w_minus_indices[-(i+1)]  # Bottom words
```

### 2. Optimized Temperature Scaling
**Before**: 0.5 (quá conservative)
**Now**: 0.3 (balance stability vs signal strength)

### 3. Enhanced DPO Loss với Confidence Weighting
**Key Innovation**: Weight loss theo confidence level
```python
confidence_weights = torch.abs(logits).detach()
confidence_weights = torch.clamp(confidence_weights, 0.5, 2.0)
losses = base_losses * confidence_weights
```

### 4. Adaptive Performance Boosting
```python
if avg_acc < 0.55:  # Poor performance
    losses = losses * 0.7  # Reduce impact
elif avg_acc > 0.8:   # Strong performance  
    losses = losses * 1.2  # Amplify signal
```

### 5. TC_15-Focused Regularization
**New Components**:
- **Coherence reg**: Maximize top-15 words probability mass
- **Controlled diversity**: Only penalize high similarity (>0.7)
- **Sparsity via entropy**: Encourage focused topics
- **Reference similarity**: Maintain ~0.8 similarity to reference

### 6. Enhanced Loss Weighting
**DPO Scale**: Allow up to 2x lambda_dpo với minimum threshold
**Goal**: Ensure DPO has meaningful impact on final loss

## Expected Improvements

1. **TC_15**: Target +0.002-0.005 improvement (to 0.442-0.445)
2. **Stability**: Better convergence với confidence weighting
3. **Signal Quality**: Stronger DPO gradients từ better pairing
4. **Coherence**: Enhanced regularization specifically cho TC_15

## Key Metrics to Monitor

1. **Reward Accuracy**: Should be >65% consistently
2. **Confidence Weights**: Monitor distribution (0.5-2.0 range)
3. **DPO Scale**: Track actual scaling factors
4. **Loss Ratios**: DPO/Total loss ratio should be 5-15%

## Success Criteria

- **Primary**: TC_15 ≥ 0.442 (minimum +0.002 improvement)
- **Stability**: Reward accuracy >65% during training
- **Coherence**: Top-15 words mass improvement
- **Balance**: No single loss component dominates

## Next Steps if Still Insufficient

If TC_15 < 0.442:
1. Increase `lambda_dpo` from 0.3 to 0.5
2. Add topic-specific preference weighting
3. Implement multi-step DPO refinement
4. Consider curriculum learning approach

Run test và monitor logs để verify improvements!

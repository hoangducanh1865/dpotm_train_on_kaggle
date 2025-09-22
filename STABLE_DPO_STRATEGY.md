# Stable DPO Fine-tuning Strategy

## Problem Analysis

Sau khi phân tích kết quả training, phát hiện **3 vấn đề chính** khiến TC_15 giảm từ 0.44019 → 0.43581:

### 1. Preference Pair Generation Không Hợp Lý
**Vấn đề**: Tạo tất cả combinations của (w_plus, w_minus), gây ra:
- Quá nhiều preference pairs không meaningful
- Training signal diluted và noisy
- Model bị confuse bởi conflicting preferences

**Giải pháp**: Chỉ tạo 10 pairs tối đa mỗi topic với pairing hợp lý

### 2. Temperature Quá Thấp Gây Instability
**Vấn đề**: `temperature = 0.1` tạo ra distributions quá sharp:
- Gradients unstable và extreme
- Overfitting to noisy preferences
- Loss landscape không smooth

**Giải pháp**: Tăng `temperature = 0.5` cho stability tốt hơn

### 3. Thiếu Gradient Control
**Vấn đề**: Không có gradient clipping trong fine-tuning:
- Large gradients destabilize training
- Parameters drift từ good initialization
- Loss components không cân bằng

**Giải pháp**: Thêm gradient clipping và stability monitoring

## Key Improvements

### 1. Smart Preference Pairing
```python
# Thay vì: for w_minus in w_minus_indices:
#              for w_plus in w_plus_indices:
# Sử dụng:
min_pairs = min(len(w_plus_indices), len(w_minus_indices))
max_pairs_per_topic = min(10, min_pairs * 2)
for i in range(max_pairs_per_topic):
    w_plus_idx = w_plus_indices[i % len(w_plus_indices)]
    w_minus_idx = w_minus_indices[i % len(w_minus_indices)]
```

### 2. Moderate Temperature Scaling
```python
temperature = 0.5  # Thay vì 0.1 cho stability
```

### 3. Gradient Clipping & Monitoring
```python
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```

### 4. Logits Clamping
```python
logits = torch.clamp(logits, -10.0, 10.0)  # Tránh extreme values
```

### 5. Adaptive Loss Scaling
```python
if avg_acc < 0.6:  # Nếu reward accuracy thấp
    losses = losses * 0.5  # Giảm magnitude
```

## Expected Results

Với các cải tiến này, dự kiến:
- **TC_15 preservation**: Không bị giảm sau fine-tuning
- **Training stability**: Smoother convergence
- **Gradient health**: Controlled parameter updates
- **Better preferences**: Meaningful DPO signal

## Success Criteria

1. **TC_15 after fine-tuning ≥ TC_15 before fine-tuning**
2. **Reward accuracy > 60%** trong quá trình training
3. **Gradient norm < 5.0** để đảm bảo stability
4. **Loss components balanced** không có component nào dominate

## Usage

Chạy training với cấu hình hiện tại:
```bash
python main.py
```

Monitor logs để kiểm tra:
- Reward accuracy
- Gradient norms
- Loss component ratios
- TC_15 progression

Nếu vẫn có vấn đề, có thể giảm `lambda_dpo` thêm hoặc tăng `temperature` lên 0.7.

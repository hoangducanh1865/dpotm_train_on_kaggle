# ECRTM với DPO Fine-tuning - Cải thiện TC_15

## Các cải tiến đã thực hiện

### 1. DPO (Direct Preference Optimization) Fine-tuning
- **Vectorized DPO loss computation**: Batch processing với 512 samples
- **IPO option**: Identity Preference Optimization cho training ổn định hơn
- **Label smoothing**: Giảm overfitting trong preference learning
- **Adaptive loss weighting**: Cân bằng các loss components tự động

### 2. Enhanced Regularization
- **Multi-component regularization**: L2 + KL divergence + entropy regularization
- **Topic diversity loss**: Khuyến khích topics khác biệt nhau
- **Adaptive scaling**: Điều chỉnh trọng số loss dựa trên magnitude

### 3. Optimized Hyperparameters
Dựa trên phân tích kết quả training trước đó:

| Parameter | Giá trị cũ | Giá trị mới | Lý do |
|-----------|------------|-------------|-------|
| `weight_ECR` | 100 | 150-200 | Tăng embedding clustering regularization |
| `weight_InfoNCE` | 50 | 100-150 | Tăng contrastive learning |
| `beta_temp` | 0.2 | 0.1-0.15 | Cải thiện topic-word distribution |
| `lambda_dpo` | - | 0.5-0.7 | DPO preference learning weight |
| `lambda_diversity` | - | 0.1-0.2 | Topic diversity regularization |

### 4. Performance Monitoring
- **Preference accuracy**: Đo độ chính xác của preference predictions
- **Reward margin**: Đo khoảng cách giữa chosen và rejected samples
- **Loss component ratios**: Theo dõi cân bằng giữa các loss terms

## Cách sử dụng

### Training đơn giản với hyperparameters được optimize:
```bash
python run_optimized_training.py --dataset 20NG
```

### Hyperparameter sweep (thử nhiều cấu hình):
```bash
python run_optimized_training.py --sweep --dataset 20NG
```

### Training thủ công với các tham số cụ thể:
```bash
python main.py --model ECRTM --dataset 20NG --num_topics 50 \
    --weight_ECR 175 --weight_InfoNCE 125 --beta_temp 0.12 \
    --lambda_dpo 0.6 --lambda_reg 0.008 --lambda_diversity 0.15 \
    --use_ipo --label_smoothing 0.05 --device cuda
```

## Các tham số mới

### DPO Fine-tuning
- `--lambda_dpo`: Trọng số cho DPO loss (default: 0.5)
- `--lambda_reg`: Trọng số cho regularization loss (default: 0.005) 
- `--lambda_diversity`: Trọng số cho topic diversity loss (default: 0.1)
- `--use_ipo`: Sử dụng IPO thay vì standard DPO
- `--label_smoothing`: Label smoothing cho preference loss (default: 0.0)

### Hyperparameters được tune
- `--weight_ECR`: Đề xuất 150-200 (default: 100)
- `--weight_InfoNCE`: Đề xuất 100-150 (default: 50)
- `--beta_temp`: Đề xuất 0.1-0.15 (default: 0.2)

## Expected Improvements

Với các cải tiến này, dự kiến TC_15 sẽ tăng từ **0.42977** lên **>0.45-0.48**:

1. **DPO fine-tuning**: +2-3% cải thiện topic coherence
2. **Topic diversity regularization**: +1-2% cải thiện topic distinctiveness  
3. **Optimized hyperparameters**: +1-2% từ better embedding clustering
4. **Enhanced regularization**: +0.5-1% từ reduced overfitting

## Monitoring Training

Trong quá trình training, theo dõi các metrics:
- `loss_DPO`: Giảm dần, stabilize around 0.1-0.3
- `preference_accuracy`: Tăng dần, target >0.6
- `reward_margin`: Tăng dần, target >0.1
- `loss_diversity`: Giảm dần, maintain >0
- `TC_15`: Target >0.45

## Troubleshooting

1. **Nếu loss exploding**: Giảm learning rate hoặc giảm lambda values
2. **Nếu TC_15 không cải thiện**: Tăng weight_ECR hoặc giảm beta_temp
3. **Nếu topics không diverse**: Tăng lambda_diversity
4. **Nếu preference accuracy thấp**: Kiểm tra preference dataset quality

## Files được modify

- `models/ECRTM/ECRTM.py`: Core model với DPO và diversity regularization
- `utils/config.py`: Thêm DPO parameters
- `main.py`: Model initialization với new parameters
- `run_optimized_training.py`: Optimized training script

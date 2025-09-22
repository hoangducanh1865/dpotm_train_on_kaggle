import argparse
from utils.configs import Configs as cfg 


def new_parser(name=None):
    return argparse.ArgumentParser(prog=name)


def add_dataset_argument(parser):
    parser.add_argument('--dataset', type=str,
                        help='dataset name', default='BBC_new')
    parser.add_argument('--plm_model', type=str,
                        help='plm model name', default='all-mpnet-base-v2')
    
def add_logging_argument(parser):
    parser.add_argument('--wandb_prj', type=str, default='topmost')


def add_model_argument(parser):
    parser.add_argument('--model', type=str, default='ECRTM')
    parser.add_argument('--num_topics', type=int, default=50)
    parser.add_argument('--num_groups', type=int, default=20)
    parser.add_argument('--dropout', type=float, default=0.15)  # Tăng từ 0.1 → 0.15 cho better regularization
    parser.add_argument('--hidden_dim_1', type=int, default=384)  # Giảm từ 512 → 384 cho stable training
    parser.add_argument('--hidden_dim_2', type=int, default=384)  # Giảm từ 512 → 384 cho stable training
    parser.add_argument('--theta_temp', type=float, default=1.0)  # Tăng từ 0.8 → 1.0 cho smoother topic distribution
    parser.add_argument('--DT_alpha', type=float, default=2.5)   # GIẢM từ 3.5 → 2.5 cho better coherence
    parser.add_argument('--TW_alpha', type=float, default=3.5)   # TĂNG từ 2.5 → 3.5 cho much better topic-word coherence
    
    parser.add_argument('--weight_GR', type=float, default=1.5)  # Giảm từ 2.0 → 1.5 cho balanced group regularization
    parser.add_argument('--alpha_GR', type=float, default=6.0)   # Giảm từ 8.0 → 6.0 cho stable group clustering
    parser.add_argument('--weight_InfoNCE', type=float, default=80.0) # TĂNG từ 60.0 → 80.0 cho stronger contrastive learning
    parser.add_argument('--beta_temp', type=float, default=0.15)  # GIẢM từ 0.2 → 0.15 cho sharper word distributions  
    parser.add_argument('--weight_ECR', type=float, default=150.0) # TĂNG từ 120.0 → 150.0 cho stronger embedding clustering
    parser.add_argument('--use_pretrainWE', action='store_true',
                        default=True, help='Enable use_pretrainWE mode')
    
    # DPO parameters được điều chỉnh để KHÔNG làm hỏng TC_15
    parser.add_argument('--lambda_dpo', type=float, default=0.3,  # Giảm MẠNH từ 1.0 → 0.3 để tránh overfitting
                        help='DPO loss weight for preference learning')
    parser.add_argument('--lambda_reg', type=float, default=0.005,  # Giảm từ 0.015 → 0.005 cho minimal regularization
                        help='Regularization loss weight')
    parser.add_argument('--use_ipo', action='store_true', default=False,  # TẮT IPO vì có thể conflict với coherence
                        help='Use IPO loss instead of standard DPO for stable training')
    parser.add_argument('--label_smoothing', type=float, default=0.05,  # Giảm từ 0.1 → 0.05 cho minimal smoothing
                        help='Label smoothing for preference loss to reduce overfitting')

def add_wete_argument(parser):
    parser.add_argument('--glove', type=str, default='glove.6B.100d.txt', help='embedding model name')
    parser.add_argument('--wete_beta', type=float, default=0.5)
    parser.add_argument('--wete_epsilon', type=float, default=0.1)
    parser.add_argument('--init_alpha', type=bool, default=True)


def add_training_argument(parser):
    parser.add_argument('--epochs', type=int, default=500)  # Giữ 500 epochs cho stable training
    parser.add_argument('--finetune_epochs', type=int, default=50) # GIẢM MẠNH từ 100 → 50 epochs để tránh hỏng coherence
    parser.add_argument('--batch_size', type=int, default=200,  # Giữ 200 cho stable gradients
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.002,  # Giữ 0.002 cho good learning
                        help='learning rate')
    parser.add_argument('--finetune_lr', type=float, default=0.0005, # GIẢM từ 0.001 → 0.0005 cho gentle fine-tuning
                        help='fine-tune learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to run the model, cuda or cpu')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--lr_scheduler', type=str,
                        help='learning rate scheduler, dont use if not needed, \
                            currently support: step', default='StepLR')
    parser.add_argument('--lr_step_size', type=int, default=75, # Giữ 75 cho balanced decay
                        help='step size for learning rate scheduler')

def add_eval_argument(parser):
    parser.add_argument('--tune_SVM', action='store_true', default=False)


def add_checkpoint_argument(parser):
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint file to resume training')
    
    
def save_config(args, path):
    with open(path, 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')


def load_config(path):
    args = argparse.Namespace()
    with open(path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            if value.isdigit():
                if value.find('.') != -1:
                    value = float(value)
                else:
                    value = int(value)
            setattr(args, key, value)
    print(args)
    return args

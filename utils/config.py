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
    parser.add_argument('--theta_temp', type=float, default=2.0)  # TĂNG CỰC MẠNH từ 1.5 → 2.0 cho extreme diversity
    parser.add_argument('--DT_alpha', type=float, default=1.2)   # GIẢM CỰC MẠNH từ 1.8 → 1.2 cho ultra-loose constraints
    parser.add_argument('--TW_alpha', type=float, default=2.0)   # GIẢM MẠNH từ 2.8 → 2.0 cho minimal coherence constraints
    
    parser.add_argument('--weight_GR', type=float, default=0.3)  # GIẢM CỰC MẠNH từ 0.8 → 0.3 cho zero clustering
    parser.add_argument('--alpha_GR', type=float, default=1.5)   # GIẢM CỰC MẠNH từ 3.0 → 1.5 cho ultra-loose clustering
    parser.add_argument('--weight_InfoNCE', type=float, default=20.0) # GIẢM CỰC MẠNH từ 40.0 → 20.0 cho minimal contrastive
    parser.add_argument('--beta_temp', type=float, default=0.55)  # TĂNG CỰC MẠNH từ 0.35 → 0.55 cho maximum diversity  
    parser.add_argument('--weight_ECR', type=float, default=50.0) # GIẢM CỰC MẠNH từ 80.0 → 50.0 để loại bỏ clustering
    parser.add_argument('--use_pretrainWE', action='store_true',
                        default=True, help='Enable use_pretrainWE mode')
    
    # DPO controls - ULTRA DIVERSITY-FOCUSED optimization for TD > 0.90
    parser.add_argument('--disable_dpo', action='store_true', default=False,
                        help='Disable DPO loss completely during fine-tuning')
    parser.add_argument('--lambda_dpo', type=float, default=0.3,  # GIẢM MẠNH for diversity focus
                        help='DPO loss weight for preference learning')
    parser.add_argument('--lambda_reg', type=float, default=0.5,  # TĂNG CỰC MẠNH regularization for diversity
                        help='Regularization loss weight')
    parser.add_argument('--use_ipo', action='store_true', default=True,  # Enable IPO for stable training
                        help='Use IPO loss instead of standard DPO for stable training')
    parser.add_argument('--label_smoothing', type=float, default=0.2,  # TĂNG smoothing for diversity
                        help='Label smoothing for preference loss to reduce overfitting')

def add_wete_argument(parser):
    parser.add_argument('--glove', type=str, default='glove.6B.100d.txt', help='embedding model name')
    parser.add_argument('--wete_beta', type=float, default=0.5)
    parser.add_argument('--wete_epsilon', type=float, default=0.1)
    parser.add_argument('--init_alpha', type=bool, default=True)


def add_training_argument(parser):
    parser.add_argument('--epochs', type=int, default=500)  # Keep 500 epochs for stable base training
    parser.add_argument('--finetune_epochs', type=int, default=200) # TĂNG to 200 for intensive diversity training
    parser.add_argument('--batch_size', type=int, default=200,  # Keep 200 for stable gradients
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.002,  # Keep base learning rate
                        help='learning rate')
    parser.add_argument('--finetune_lr', type=float, default=0.001, # TĂNG CỰC MẠNH for aggressive diversity training
                        help='fine-tune learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to run the model, cuda or cpu')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--lr_scheduler', type=str,
                        help='learning rate scheduler, dont use if not needed, \
                            currently support: step', default='StepLR')
    parser.add_argument('--lr_step_size', type=int, default=200, # Match finetune_epochs for consistent training
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

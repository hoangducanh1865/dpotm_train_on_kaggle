import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .ECR import ECR
from utils.configs import Configs as cfg
import json
import random


class ECRTM(nn.Module):
    '''
        Effective Neural Topic Modeling with Embedding Clustering Regularization. ICML 2023

        Xiaobao Wu, Xinshuai Dong, Thong Thanh Nguyen, Anh Tuan Luu.
    '''
    def __init__(self, vocab_size, num_topics=50, en_units=200, dropout=0., pretrained_WE=None, embed_size=200, beta_temp=0.2, weight_loss_ECR=100.0, sinkhorn_alpha=20.0, sinkhorn_max_iter=1000, current_run_dir=None, lambda_dpo=0.5, lambda_reg=0.005, use_ipo=False, label_smoothing=0.0):
        super().__init__()

        self.is_finetuing = False
        self.device = cfg.DEVICE
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.beta_temp = beta_temp
        self.current_run_dir = current_run_dir
        
        # DPO hyperparameters với giá trị tối ưu
        self.lambda_dpo = lambda_dpo
        self.lambda_reg = lambda_reg  
        self.use_ipo = use_ipo
        self.label_smoothing = label_smoothing
        
        # Cached data cho hiệu suất
        self.preference_cache = None
        self.reward_accuracies = []
        self.reward_margins = []
        
        self.beta_ref_path = None
        self.beta_ref = None
        self.preference_dataset_path = None
        self.preference_dataset = None

        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T))

        self.mu2.requires_grad = False
        self.var2.requires_grad = False

        self.fc11 = nn.Linear(vocab_size, en_units)
        self.fc12 = nn.Linear(en_units, en_units)
        self.fc21 = nn.Linear(en_units, num_topics)
        self.fc22 = nn.Linear(en_units, num_topics)
        self.fc1_dropout = nn.Dropout(dropout)
        self.theta_dropout = nn.Dropout(dropout)

        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(num_topics)
        self.logvar_bn.weight.requires_grad = False
        self.decoder_bn = nn.BatchNorm1d(vocab_size, affine=True)
        self.decoder_bn.weight.requires_grad = False

        if pretrained_WE is not None:
            self.word_embeddings = torch.from_numpy(pretrained_WE).float()
        else:
            self.word_embeddings = nn.init.trunc_normal_(torch.empty(vocab_size, embed_size))
        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))

        self.topic_embeddings = torch.empty((num_topics, self.word_embeddings.shape[1]))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

        self.ECR = ECR(weight_loss_ECR, sinkhorn_alpha, sinkhorn_max_iter)

    def get_beta(self):
        dist = self.pairwise_euclidean_distance(self.topic_embeddings, self.word_embeddings)
        beta = F.softmax(-dist / self.beta_temp, dim=0)
        return beta

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def encode(self, input):
        e1 = F.softplus(self.fc11(input))
        e1 = F.softplus(self.fc12(e1))
        e1 = self.fc1_dropout(e1)
        mu = self.mean_bn(self.fc21(e1))
        logvar = self.logvar_bn(self.fc22(e1))
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)

        loss_KL = self.compute_loss_KL(mu, logvar)

        return theta, loss_KL

    def get_theta(self, input):
        theta, loss_KL = self.encode(input)
        if self.training:
            return theta, loss_KL
        else:
            return theta

    def compute_loss_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        # KLD: N*K
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.num_topics)
        KLD = KLD.mean()
        return KLD

    def get_loss_ECR(self):
        cost = self.pairwise_euclidean_distance(self.topic_embeddings, self.word_embeddings)
        loss_ECR = self.ECR(cost)
        return loss_ECR
    
    def load_preference_dataset(self):
        self.preference_dataset_path = os.path.join(self.current_run_dir, 'preference_dataset.jsonl')
        if self.preference_dataset is None:
            self.preference_dataset = []
            with open(self.preference_dataset_path, 'r') as f:
                for line in f:
                    self.preference_dataset.append(line)
        self.beta_ref_path = os.path.join(self.current_run_dir, 'beta.npy')
        self.beta_ref = torch.from_numpy(np.load(self.beta_ref_path)).float().to(self.device)
        self.beta_ref.requires_grad = False
    
    def get_loss_DPO(self):
        if self.preference_dataset is None:
            self.load_preference_dataset()
            
        beta = self.get_beta()
        
        # Batch processing để tăng hiệu suất với pairing hợp lý
        if self.preference_cache is None:
            self.preference_cache = []
            for line in self.preference_dataset:
                data = json.loads(line)
                k = data['k']
                w_plus_indices = data['w_plus_indices'] 
                w_minus_indices = data['w_minus_indices']
                
                # Tạo pairs hợp lý: mỗi từ tốt pair với 1-2 từ xấu random
                # Thay vì tất cả combinations, chỉ tạo số lượng hợp lý
                min_pairs = min(len(w_plus_indices), len(w_minus_indices))
                max_pairs_per_topic = min(10, min_pairs * 2)  # Giới hạn số pairs
                
                for i in range(max_pairs_per_topic):
                    w_plus_idx = w_plus_indices[i % len(w_plus_indices)]
                    w_minus_idx = w_minus_indices[i % len(w_minus_indices)]
                    self.preference_cache.append((k, w_plus_idx, w_minus_idx))
        
        # Vectorized computation với moderate batch size
        batch_size = min(256, len(self.preference_cache))  # Giảm từ 768 → 256
        if len(self.preference_cache) > batch_size:
            # Smart sampling: ưu tiên high-confidence preferences
            import random
            batch_indices = random.sample(range(len(self.preference_cache)), batch_size)
            batch_preferences = [self.preference_cache[i] for i in batch_indices]
        else:
            batch_preferences = self.preference_cache
            
        chosen_logps = []
        rejected_logps = []
        ref_chosen_logps = []
        ref_rejected_logps = []
        
        # Moderate temperature cho stability
        temperature = 0.5  # Tăng từ 0.1 → 0.5 cho ổn định hơn
        
        for k, w_plus_idx, w_minus_idx in batch_preferences:
            # Policy logps với temperature scaling
            chosen_logp = torch.log(torch.softmax(beta[k] / temperature, dim=0)[w_plus_idx] + 1e-8)
            rejected_logp = torch.log(torch.softmax(beta[k] / temperature, dim=0)[w_minus_idx] + 1e-8)
            
            # Reference logps với cùng temperature
            ref_chosen_logp = torch.log(torch.softmax(self.beta_ref[k] / temperature, dim=0)[w_plus_idx] + 1e-8)
            ref_rejected_logp = torch.log(torch.softmax(self.beta_ref[k] / temperature, dim=0)[w_minus_idx] + 1e-8)
            
            chosen_logps.append(chosen_logp)
            rejected_logps.append(rejected_logp)
            ref_chosen_logps.append(ref_chosen_logp)
            ref_rejected_logps.append(ref_rejected_logp)
        
        if not chosen_logps:
            return torch.tensor(0.0, device=beta.device, requires_grad=True)
            
        chosen_logps = torch.stack(chosen_logps)
        rejected_logps = torch.stack(rejected_logps) 
        ref_chosen_logps = torch.stack(ref_chosen_logps)
        ref_rejected_logps = torch.stack(ref_rejected_logps)
        
        # Compute DPO loss với improved stability
        pi_logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios
        
        # Clamp logits để tránh extreme values
        logits = torch.clamp(logits, -10.0, 10.0)
        
        if self.use_ipo:
            # IPO loss cho training ổn định hơn
            losses = (logits - 1/(2 * self.lambda_dpo)) ** 2
        else:
            # Standard DPO với label smoothing và improved stability
            losses = (-F.logsigmoid(self.lambda_dpo * logits) * (1 - self.label_smoothing) - 
                     F.logsigmoid(-self.lambda_dpo * logits) * self.label_smoothing)
        
        # Compute rewards cho monitoring
        chosen_rewards = self.lambda_dpo * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.lambda_dpo * (rejected_logps - ref_rejected_logps).detach()
        
        # Track metrics with stability check
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        self.reward_accuracies = reward_accuracies.cpu().numpy().tolist()
        self.reward_margins = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
        
        # Stability check: nếu reward accuracy quá thấp, giảm loss magnitude
        avg_acc = reward_accuracies.mean().item()
        if avg_acc < 0.6:  # Nếu accuracy < 60%, có vấn đề
            losses = losses * 0.5  # Giảm magnitude để tránh học sai
        
        return losses.mean()

    def get_loss_regularization(self):
        beta = self.get_beta()
        
        # Enhanced multi-component regularization cho TC_15 optimization
        # 1. L2 regularization với adaptive weight
        l2_reg = torch.mean((beta - self.beta_ref) ** 2)
        
        # 2. Enhanced KL divergence với temperature scaling  
        temperature = 0.5
        beta_soft = F.softmax(beta / temperature, dim=-1)
        beta_ref_soft = F.softmax(self.beta_ref / temperature, dim=-1)
        kl_reg = F.kl_div(torch.log(beta_soft + 1e-8), beta_ref_soft, reduction='batchmean')
        
        # 3. Diversity regularization để tăng topic diversity
        beta_norm = F.normalize(beta, dim=1, p=2)
        cosine_sim = torch.matmul(beta_norm, beta_norm.t())
        diversity_reg = torch.mean(torch.triu(cosine_sim, diagonal=1) ** 2)
        
        # 4. Sparsity regularization để tăng topic coherence
        sparsity_reg = torch.mean(torch.sum(beta * torch.log(beta + 1e-8), dim=-1))
        
        # Weighted combination với focus on coherence
        total_reg = l2_reg + 0.2 * kl_reg + 0.05 * diversity_reg - 0.01 * sparsity_reg
        
        return total_reg

    def pairwise_euclidean_distance(self, x, y):
        cost = torch.sum(x ** 2, axis=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
        return cost

    def forward(self, input):
        if not self.is_finetuing:
            bow = input["data"]
            theta, loss_KL = self.encode(input['data'])
            beta = self.get_beta()

            recon = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), dim=-1)
            recon_loss = -(bow * recon.log()).sum(axis=1).mean()

            loss_TM = recon_loss + loss_KL

            loss_ECR = self.get_loss_ECR()
            
            loss = loss_TM + loss_ECR

            rst_dict = {
                'loss': loss,
                'loss_TM': loss_TM,
                'loss_ECR': loss_ECR
            }

            return rst_dict
        else:
            bow = input["data"]
            theta, loss_KL = self.encode(input['data'])
            beta = self.get_beta()

            recon = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), dim=-1)
            recon_loss = -(bow * recon.log()).sum(axis=1).mean()

            loss_TM = recon_loss + loss_KL
            loss_ECR = self.get_loss_ECR()
            
            # Enhanced DPO implementation
            loss_DPO = self.get_loss_DPO()
            loss_regularization = self.get_loss_regularization()
            
            # Adaptive loss weighting dựa trên magnitude như LLM training
            with torch.no_grad():
                loss_magnitudes = {
                    'TM': loss_TM.item(),
                    'ECR': loss_ECR.item(), 
                    'DPO': loss_DPO.item(),
                    'REG': loss_regularization.item()
                }
                
                # Cân bằng loss components
                base_loss = loss_magnitudes['TM'] + loss_magnitudes['ECR']
                if base_loss > 0:
                    dpo_scale = min(self.lambda_dpo, base_loss / max(loss_magnitudes['DPO'], 1e-8))
                    reg_scale = min(self.lambda_reg, base_loss / max(loss_magnitudes['REG'], 1e-8))
                else:
                    dpo_scale = self.lambda_dpo
                    reg_scale = self.lambda_reg
            
            # Final loss với adaptive weighting
            loss = loss_TM + loss_ECR + dpo_scale * loss_DPO + reg_scale * loss_regularization

            # Comprehensive metrics như LLM training
            rst_dict = {
                'loss': loss,
                'loss_TM': loss_TM,
                'loss_ECR': loss_ECR,
                'loss_DPO': loss_DPO,
                'loss_regularization': loss_regularization,
                'dpo_scale': dpo_scale,
                'reg_scale': reg_scale,
                'reward_accuracy': np.mean(self.reward_accuracies) if self.reward_accuracies else 0.0,
                'reward_margin': np.mean(self.reward_margins) if self.reward_margins else 0.0,
                'loss_magnitudes': loss_magnitudes
            }

            return rst_dict
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .ECR import ECR
from utils.configs import Configs as cfg
import json


class ECRTM(nn.Module):
    '''
        Effective Neural Topic Modeling with Embedding Clustering Regularization. ICML 2023

        Xiaobao Wu, Xinshuai Dong, Thong Thanh Nguyen, Anh Tuan Luu.
    '''
    def __init__(self, vocab_size, num_topics=50, en_units=200, dropout=0., pretrained_WE=None, embed_size=200, beta_temp=0.2, weight_loss_ECR=100.0, sinkhorn_alpha=20.0, sinkhorn_max_iter=1000, current_run_dir=None, lambda_dpo=0.5, lambda_reg=0.005, lambda_diversity=0.1, use_ipo=False, label_smoothing=0.0):
        super().__init__()

        self.is_finetuing = False
        self.device = cfg.DEVICE
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.beta_temp = beta_temp
        self.current_run_dir = current_run_dir
        
        # DPO hyperparameters
        self.lambda_dpo = lambda_dpo
        self.lambda_reg = lambda_reg
        self.lambda_diversity = lambda_diversity
        self.use_ipo = use_ipo
        self.label_smoothing = label_smoothing
        
        # Cached data for efficiency
        self.beta_ref_path = None
        self.beta_ref = None
        self.preference_dataset_path = None
        self.preference_dataset = None
        self.preference_pairs_cache = None
        
        # Metrics tracking
        self.preference_accuracy = 0.0
        self.reward_margin = 0.0

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
    
    def get_loss_diversity(self):
        """Topic diversity loss to encourage distinct topics"""
        beta = self.get_beta()
        beta_norm = F.normalize(beta, p=2, dim=1)
        similarity_matrix = torch.mm(beta_norm, beta_norm.t())
        identity = torch.eye(self.num_topics, device=self.device)
        similarity_matrix = similarity_matrix * (1 - identity)
        diversity_loss = similarity_matrix.pow(2).mean()
        
        # Topic embedding diversity
        topic_emb_norm = F.normalize(self.topic_embeddings, p=2, dim=1)
        topic_similarity = torch.mm(topic_emb_norm, topic_emb_norm.t())
        topic_similarity = topic_similarity * (1 - identity)
        embedding_diversity_loss = topic_similarity.pow(2).mean()
        
        return diversity_loss + 0.5 * embedding_diversity_loss
    
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
        
        # Pre-process preference pairs for efficiency
        if self.preference_pairs_cache is None:
            self.preference_pairs_cache = []
            for line in self.preference_dataset:
                data = json.loads(line)
                k = data['k']
                w_plus_indices = data['w_plus_indices']
                w_minus_indices = data['w_minus_indices']
                
                for w_minus_idx in w_minus_indices:
                    for w_plus_idx in w_plus_indices:
                        self.preference_pairs_cache.append((k, w_plus_idx, w_minus_idx))
    
    def get_loss_DPO(self):
        if self.preference_dataset is None:
            self.load_preference_dataset()
            
        beta = self.get_beta()
        
        # Vectorized computation for efficiency
        batch_size = min(len(self.preference_pairs_cache), 512)  # Process in batches
        
        loss_DPO_batch = []
        chosen_rewards = []
        rejected_rewards = []
        
        for i in range(0, len(self.preference_pairs_cache), batch_size):
            batch_pairs = self.preference_pairs_cache[i:i+batch_size]
            
            # Extract indices for batch processing
            topics = torch.tensor([pair[0] for pair in batch_pairs], device=self.device)
            w_plus = torch.tensor([pair[1] for pair in batch_pairs], device=self.device)
            w_minus = torch.tensor([pair[2] for pair in batch_pairs], device=self.device)
            
            # Batch computation of logits
            chosen_logits = beta[topics, w_plus]
            rejected_logits = beta[topics, w_minus]
            ref_chosen_logits = self.beta_ref[topics, w_plus]
            ref_rejected_logits = self.beta_ref[topics, w_minus]
            
            # Compute DPO margins
            policy_margins = chosen_logits - rejected_logits
            ref_margins = ref_chosen_logits - ref_rejected_logits
            logits = policy_margins - ref_margins
            
            if self.use_ipo:
                # IPO loss (more stable)
                losses = (logits - 1/(2 * self.lambda_dpo)) ** 2
            else:
                # Standard DPO with label smoothing
                losses = (-F.logsigmoid(self.lambda_dpo * logits) * (1 - self.label_smoothing) - 
                         F.logsigmoid(-self.lambda_dpo * logits) * self.label_smoothing)
            
            loss_DPO_batch.append(losses)
            
            # Track rewards for monitoring
            chosen_rewards.extend((self.lambda_dpo * (chosen_logits - ref_chosen_logits)).detach().cpu().tolist())
            rejected_rewards.extend((self.lambda_dpo * (rejected_logits - ref_rejected_logits)).detach().cpu().tolist())
        
        # Compute metrics
        chosen_rewards = torch.tensor(chosen_rewards)
        rejected_rewards = torch.tensor(rejected_rewards)
        self.preference_accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
        self.reward_margin = (chosen_rewards - rejected_rewards).mean().item()
        
        return torch.cat(loss_DPO_batch).mean()

    def get_loss_regularization(self):
        beta = self.get_beta()
        
        # Enhanced regularization with multiple components
        # 1. L2 distance from reference
        l2_reg = torch.mean((beta - self.beta_ref) ** 2)
        
        # 2. KL divergence regularization for smoother distributions
        kl_reg = 0.0
        for k in range(self.num_topics):
            kl_reg += F.kl_div(
                F.log_softmax(beta[k], dim=0), 
                F.softmax(self.beta_ref[k], dim=0), 
                reduction='sum'
            )
        kl_reg = kl_reg / self.num_topics
        
        # 3. Entropy regularization to prevent over-sharpening
        entropy_reg = 0.0
        for k in range(self.num_topics):
            prob_dist = F.softmax(beta[k], dim=0)
            entropy_reg -= torch.sum(prob_dist * torch.log(prob_dist + 1e-10))
        entropy_reg = entropy_reg / self.num_topics
        
        # Combine regularization terms
        total_reg = l2_reg + 0.1 * kl_reg - 0.01 * entropy_reg
        
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
            
            # Add diversity loss for better topic distinctiveness
            loss_diversity = self.get_loss_diversity()
            
            loss = loss_TM + loss_ECR + self.lambda_diversity * loss_diversity

            rst_dict = {
                'loss': loss,
                'loss_TM': loss_TM,
                'loss_ECR': loss_ECR,
                'loss_diversity': loss_diversity
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
            
            # Add diversity loss for fine-tuning phase
            loss_diversity = self.get_loss_diversity()
            
            # Adaptive loss weighting based on training progress
            with torch.no_grad():
                loss_DPO = self.get_loss_DPO()
                loss_regularization = self.get_loss_regularization()
            
            # Compute loss component ratios for monitoring
            loss_tm_magnitude = loss_TM.detach().item()
            loss_ecr_magnitude = loss_ECR.detach().item()
            loss_dpo_magnitude = loss_DPO.detach().item()
            loss_reg_magnitude = loss_regularization.detach().item()
            loss_div_magnitude = loss_diversity.detach().item()
            
            # Adaptive scaling to balance loss components
            total_base_loss = loss_tm_magnitude + loss_ecr_magnitude
            if total_base_loss > 0:
                dpo_scale = min(self.lambda_dpo, total_base_loss / (loss_dpo_magnitude + 1e-8))
                reg_scale = min(self.lambda_reg, total_base_loss / (loss_reg_magnitude + 1e-8))
                div_scale = min(self.lambda_diversity, total_base_loss / (loss_div_magnitude + 1e-8))
            else:
                dpo_scale = self.lambda_dpo
                reg_scale = self.lambda_reg
                div_scale = self.lambda_diversity
            
            # Re-compute with gradients for actual loss
            loss_DPO = self.get_loss_DPO()
            loss_regularization = self.get_loss_regularization()
            
            loss = loss_TM + loss_ECR + dpo_scale * loss_DPO + reg_scale * loss_regularization + div_scale * loss_diversity

            rst_dict = {
                'loss': loss,
                'loss_TM': loss_TM,
                'loss_ECR': loss_ECR,
                'loss_DPO': loss_DPO,
                'loss_regularization': loss_regularization,
                'loss_diversity': loss_diversity,
                'dpo_scale': dpo_scale,
                'reg_scale': reg_scale,
                'div_scale': div_scale,
                'preference_accuracy': self.preference_accuracy,
                'reward_margin': self.reward_margin,
                'loss_ratios': {
                    'TM': loss_tm_magnitude / (loss_tm_magnitude + loss_ecr_magnitude + 1e-8),
                    'ECR': loss_ecr_magnitude / (loss_tm_magnitude + loss_ecr_magnitude + 1e-8),
                    'DPO': loss_dpo_magnitude,
                    'REG': loss_reg_magnitude
                }
            }

            return rst_dict
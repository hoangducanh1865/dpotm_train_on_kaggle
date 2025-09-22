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
    def __init__(self, vocab_size, num_topics=50, en_units=200, dropout=0., pretrained_WE=None, embed_size=200, beta_temp=0.2, weight_loss_ECR=100.0, sinkhorn_alpha=20.0, sinkhorn_max_iter=1000, current_run_dir=None):
        super().__init__()

        self.is_finetuing = False
        self.device = cfg.DEVICE
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.beta_temp = beta_temp
        self.current_run_dir = current_run_dir
        
        # DPO hyperparameters - optimized for better TC_15
        self.dpo_beta = 0.1  # Lower beta for more stable training
        self.label_smoothing = 0.05  # Conservative smoothing
        self.use_ipo = True  # Use IPO for stability 
        self.reference_free = False
        
        # Enhanced preference learning parameters
        self.lambda_dpo = 0.8  # Increased DPO weight
        self.lambda_reg = 0.01  # Increased regularization
        self.lambda_diversity = 0.1  # Topic diversity weight
        self.lambda_coherence = 0.05  # Topic coherence weight
        
        # Cached data
        self.beta_ref_path = None
        self.beta_ref = None
        self.preference_dataset_path = None
        self.preference_dataset = None
        self.preference_pairs_cache = None
        
        # Metrics tracking
        self.preference_accuracy = 0.0
        self.reward_margin = 0.0
        self.chosen_rewards_sum = 0.0
        self.rejected_rewards_sum = 0.0

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
        
        # Pre-cache preference pairs for efficient batch processing
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
    
    def preference_loss_advanced(self, policy_chosen_logps, policy_rejected_logps, 
                                reference_chosen_logps, reference_rejected_logps):
        """Advanced preference loss with IPO option and label smoothing"""
        
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        
        if self.reference_free:
            ref_logratios = 0
            
        logits = pi_logratios - ref_logratios
        
        if self.use_ipo:
            # IPO loss for more stable training
            losses = (logits - 1/(2 * self.dpo_beta)) ** 2
        else:
            # Standard DPO with label smoothing
            losses = (-F.logsigmoid(self.dpo_beta * logits) * (1 - self.label_smoothing) - 
                     F.logsigmoid(-self.dpo_beta * logits) * self.label_smoothing)
        
        # Compute rewards
        chosen_rewards = self.dpo_beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.dpo_beta * (policy_rejected_logps - reference_rejected_logps).detach()
        
        return losses, chosen_rewards, rejected_rewards
    
    def get_loss_DPO(self):
        """Enhanced DPO loss with batch processing and advanced techniques"""
        if self.preference_dataset is None:
            self.load_preference_dataset()
            
        beta = self.get_beta()
        
        # Batch processing for efficiency  
        batch_size = min(512, len(self.preference_pairs_cache))
        
        if len(self.preference_pairs_cache) <= batch_size:
            pairs_to_process = self.preference_pairs_cache
        else:
            # Random sampling for diverse training
            indices = torch.randperm(len(self.preference_pairs_cache))[:batch_size]
            pairs_to_process = [self.preference_pairs_cache[i] for i in indices]
        
        policy_chosen_logps = []
        policy_rejected_logps = []
        reference_chosen_logps = []
        reference_rejected_logps = []
        
        for k, w_plus_idx, w_minus_idx in pairs_to_process:
            # Policy model probabilities
            policy_chosen_logps.append(beta[k, w_plus_idx])
            policy_rejected_logps.append(beta[k, w_minus_idx])
            
            # Reference model probabilities
            reference_chosen_logps.append(self.beta_ref[k, w_plus_idx])
            reference_rejected_logps.append(self.beta_ref[k, w_minus_idx])
        
        policy_chosen_logps = torch.stack(policy_chosen_logps)
        policy_rejected_logps = torch.stack(policy_rejected_logps)
        reference_chosen_logps = torch.stack(reference_chosen_logps)
        reference_rejected_logps = torch.stack(reference_rejected_logps)
        
        # Apply advanced preference loss
        losses, chosen_rewards, rejected_rewards = self.preference_loss_advanced(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps
        )
        
        # Update metrics
        with torch.no_grad():
            reward_accuracies = (chosen_rewards > rejected_rewards).float()
            self.preference_accuracy = reward_accuracies.mean().item()
            self.reward_margin = (chosen_rewards - rejected_rewards).mean().item()
            self.chosen_rewards_sum = chosen_rewards.mean().item()
            self.rejected_rewards_sum = rejected_rewards.mean().item()
        
        return losses.mean()
    
    def get_loss_coherence_regularization(self):
        """Topic coherence regularization to improve TC_15"""
        beta = self.get_beta()  # topics x vocab
        
        coherence_loss = 0.0
        num_top_words = 15
        
        for k in range(self.num_topics):
            # Get top words for this topic
            top_word_probs, top_word_indices = torch.topk(beta[k], k=num_top_words)
            
            # Encourage higher probability for top words
            prob_concentration = -torch.log(top_word_probs + 1e-8).mean()
            coherence_loss += prob_concentration
            
            # Encourage semantic coherence using word embeddings
            if hasattr(self, 'word_embeddings'):
                top_word_embs = self.word_embeddings[top_word_indices]
                top_word_embs_norm = F.normalize(top_word_embs, p=2, dim=1)
                
                # Compute pairwise similarities
                similarities = torch.mm(top_word_embs_norm, top_word_embs_norm.t())
                
                # Weight by word probabilities
                prob_weights = top_word_probs.unsqueeze(0) * top_word_probs.unsqueeze(1)
                weighted_similarities = similarities * prob_weights
                
                # Encourage high similarity between top words
                coherence_loss -= weighted_similarities.mean()
        
        return coherence_loss / self.num_topics
    
    def get_loss_diversity(self):
        """Topic diversity loss to ensure distinct topics"""
        beta = self.get_beta()
        
        # Normalize topic distributions
        beta_norm = F.normalize(beta, p=2, dim=1)
        
        # Compute topic similarity matrix
        similarity_matrix = torch.mm(beta_norm, beta_norm.t())
        
        # Remove diagonal (self-similarity)
        identity = torch.eye(self.num_topics, device=self.device)
        off_diagonal_similarities = similarity_matrix * (1 - identity)
        
        # Penalize high similarity between different topics
        diversity_loss = off_diagonal_similarities.pow(2).mean()
        
        # Additional topic embedding diversity
        if hasattr(self, 'topic_embeddings'):
            topic_emb_norm = F.normalize(self.topic_embeddings, p=2, dim=1)
            topic_similarity = torch.mm(topic_emb_norm, topic_emb_norm.t())
            topic_similarity = topic_similarity * (1 - identity)
            embedding_diversity_loss = topic_similarity.pow(2).mean()
            diversity_loss += 0.5 * embedding_diversity_loss
        
        return diversity_loss

    def get_loss_regularization(self):
        """Enhanced regularization combining multiple techniques"""
        beta = self.get_beta()
        
        # L2 regularization on beta deviation from reference
        l2_reg = torch.mean((beta - self.beta_ref) ** 2)
        
        # KL divergence regularization
        kl_reg = 0.0
        for k in range(self.num_topics):
            kl_div = F.kl_div(
                beta[k].log_softmax(dim=0), 
                self.beta_ref[k].softmax(dim=0), 
                reduction='sum'
            )
            kl_reg += kl_div
        kl_reg = kl_reg / self.num_topics
        
        # Entropy regularization to prevent collapse
        entropy_reg = 0.0
        for k in range(self.num_topics):
            probs = beta[k].softmax(dim=0)
            entropy = -(probs * torch.log(probs + 1e-8)).sum()
            entropy_reg -= entropy  # Negative because we want to maximize entropy
        entropy_reg = entropy_reg / self.num_topics
        
        return l2_reg + 0.1 * kl_reg + 0.01 * entropy_reg

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
            
            # Enhanced DPO fine-tuning with multiple loss components
            loss_DPO = self.get_loss_DPO()
            loss_regularization = self.get_loss_regularization()
            loss_diversity = self.get_loss_diversity()
            loss_coherence = self.get_loss_coherence_regularization()
            
            # Adaptive loss weighting for stability
            with torch.no_grad():
                loss_tm_mag = loss_TM.item()
                loss_ecr_mag = loss_ECR.item()
                loss_dpo_mag = loss_DPO.item()
                loss_reg_mag = loss_regularization.item()
                loss_div_mag = loss_diversity.item()
                loss_coh_mag = loss_coherence.item()
                
                # Normalize weights based on loss magnitudes
                total_base_loss = loss_tm_mag + loss_ecr_mag
                if total_base_loss > 0:
                    dpo_scale = min(self.lambda_dpo, total_base_loss / (loss_dpo_mag + 1e-8))
                    reg_scale = min(self.lambda_reg, total_base_loss / (loss_reg_mag + 1e-8))
                    div_scale = min(self.lambda_diversity, total_base_loss / (loss_div_mag + 1e-8))
                    coh_scale = min(self.lambda_coherence, total_base_loss / (loss_coh_mag + 1e-8))
                else:
                    dpo_scale = self.lambda_dpo
                    reg_scale = self.lambda_reg  
                    div_scale = self.lambda_diversity
                    coh_scale = self.lambda_coherence
            
            loss = (loss_TM + loss_ECR + 
                   dpo_scale * loss_DPO + 
                   reg_scale * loss_regularization +
                   div_scale * loss_diversity +
                   coh_scale * loss_coherence)

            rst_dict = {
                'loss': loss,
                'loss_TM': loss_TM,
                'loss_ECR': loss_ECR,
                'loss_DPO': loss_DPO,
                'loss_regularization': loss_regularization,
                'loss_diversity': loss_diversity,
                'loss_coherence': loss_coherence,
                # DPO metrics
                'preference_accuracy': self.preference_accuracy,
                'reward_margin': self.reward_margin,
                'chosen_rewards': self.chosen_rewards_sum,
                'rejected_rewards': self.rejected_rewards_sum,
                # Loss scaling factors
                'dpo_scale': dpo_scale,
                'reg_scale': reg_scale,
                'div_scale': div_scale,
                'coh_scale': coh_scale,
                # Loss magnitudes for monitoring
                'loss_ratios': {
                    'tm_ratio': loss_tm_mag / (total_base_loss + 1e-8),
                    'ecr_ratio': loss_ecr_mag / (total_base_loss + 1e-8),
                    'dpo_ratio': loss_dpo_mag / (total_base_loss + 1e-8),
                    'reg_ratio': loss_reg_mag / (total_base_loss + 1e-8)
                }
            }

            return rst_dict
    
    def get_optimizer(self, lr=2e-3, weight_decay=1e-4):
        """
        Enhanced optimizer with advanced learning rate scheduling and regularization
        """
        from torch.optim import AdamW
        
        # Use AdamW for better weight decay regularization
        optimizer = AdamW(
            [
                {"params": self.encoder.parameters(), "lr": lr, "weight_decay": weight_decay},
                {"params": self.fc_mean.parameters(), "lr": lr, "weight_decay": weight_decay},
                {"params": self.fc_logvar.parameters(), "lr": lr, "weight_decay": weight_decay},
                {"params": self.fc_decoder.parameters(), "lr": lr * 0.8, "weight_decay": weight_decay * 0.5},  # Slightly lower for decoder
                {"params": self.beta.parameters(), "lr": lr * 1.2, "weight_decay": weight_decay * 0.3},  # Higher for topic words
                {"params": self.w_centeroids.parameters(), "lr": lr * 0.5, "weight_decay": weight_decay * 2.0}  # Lower for centroids with more regularization
            ],
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=True  # More stable convergence
        )
        
        return optimizer
    
    def apply_gradient_clipping(self, max_norm=1.0):
        """
        Apply advanced gradient clipping with per-parameter group normalization
        """
        import torch.nn.utils
        
        # Clip gradients with different norms for different parameter groups
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm)
        torch.nn.utils.clip_grad_norm_(self.fc_mean.parameters(), max_norm)
        torch.nn.utils.clip_grad_norm_(self.fc_logvar.parameters(), max_norm * 0.8)  # More conservative for variance
        torch.nn.utils.clip_grad_norm_(self.fc_decoder.parameters(), max_norm * 1.2)
        torch.nn.utils.clip_grad_norm_(self.beta.parameters(), max_norm * 1.5)  # Allow larger gradients for topic words
        torch.nn.utils.clip_grad_norm_(self.w_centeroids.parameters(), max_norm * 0.6)  # Conservative for centroids
    
    def get_learning_rate_scheduler(self, optimizer, total_steps, warmup_steps=None):
        """
        Advanced learning rate scheduler with warmup and cosine annealing
        """
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        
        if warmup_steps is None:
            warmup_steps = min(total_steps // 10, 1000)  # 10% warmup or max 1000 steps
        
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=warmup_steps
        )
        
        # Cosine annealing scheduler
        cosine_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=total_steps - warmup_steps,
            eta_min=1e-6
        )
        
        # Combine warmup and cosine annealing
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        return scheduler
    
    def get_metrics_for_logging(self):
        """
        Get comprehensive metrics for monitoring training progress
        """
        metrics = {}
        
        # Parameter norms
        for name, param in self.named_parameters():
            if param.grad is not None:
                metrics[f'grad_norm/{name}'] = param.grad.norm().item()
            metrics[f'param_norm/{name}'] = param.norm().item()
        
        # Topic quality metrics
        beta = self.get_beta()
        if beta is not None:
            # Topic word concentration
            topic_concentration = torch.sum(beta * torch.log(beta + 1e-8), dim=1).mean()
            metrics['topic_concentration'] = topic_concentration.item()
            
            # Topic diversity within each topic
            topic_entropy = -torch.sum(beta * torch.log(beta + 1e-8), dim=1).mean()
            metrics['topic_entropy'] = topic_entropy.item()
            
            # Inter-topic similarity (lower is better for diversity)
            beta_norm = F.normalize(beta, p=2, dim=1)
            topic_similarity = torch.matmul(beta_norm, beta_norm.t())
            # Remove diagonal (self-similarity)
            mask = ~torch.eye(topic_similarity.size(0), dtype=torch.bool, device=topic_similarity.device)
            avg_similarity = topic_similarity[mask].mean()
            metrics['inter_topic_similarity'] = avg_similarity.item()
        
        return metrics
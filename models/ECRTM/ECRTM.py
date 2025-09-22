import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .ECR import ECR
from utils.configs import Configs as cfg
from utils.topic_drift_monitor import TopicDriftMonitor
from utils.preference_dataset_creator import PreferenceDatasetCreator
import json
import random


class ECRTM(nn.Module):
    '''
        Effective Neural Topic Modeling with Embedding Clustering Regularization. ICML 2023

        Xiaobao Wu, Xinshuai Dong, Thong Thanh Nguyen, Anh Tuan Luu.
    '''
    def __init__(self, vocab_size, num_topics=50, en_units=200, dropout=0., pretrained_WE=None, embed_size=200, beta_temp=0.2, weight_loss_ECR=100.0, sinkhorn_alpha=20.0, sinkhorn_max_iter=1000, current_run_dir=None, lambda_dpo=0.5, lambda_reg=0.005, use_ipo=False, label_smoothing=0.0, vocab=None):
        super().__init__()

        self.is_finetuing = False
        self.device = cfg.DEVICE
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.beta_temp = beta_temp
        self.current_run_dir = current_run_dir
        self.vocab = vocab  # Store vocab for drift monitoring
        
        # DPO hyperparameters với giá trị tối ưu
        self.lambda_dpo = lambda_dpo
        self.lambda_reg = lambda_reg  
        self.use_ipo = use_ipo
        self.label_smoothing = label_smoothing
        
        # Cached data cho hiệu suất
        self.preference_cache = None
        self.reward_accuracies = []
        self.reward_margins = []
        
        # Topic drift monitoring
        self.drift_monitor = None
        self.preference_refresh_needed = False
        
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
    
    def initialize_drift_monitoring(self):
        """Initialize topic drift monitor for preference dataset refresh."""
        if self.vocab is not None and self.drift_monitor is None:
            self.drift_monitor = TopicDriftMonitor(
                vocab=self.vocab,
                num_topics=self.num_topics,
                top_k=25,  # Monitor top-25 words
                jaccard_threshold=0.6,  # Refresh if mean Jaccard < 0.6
                check_interval=10  # Check every 10 epochs
            )
            print(f"Initialized topic drift monitor with Jaccard threshold {0.6}")
    
    def check_topic_drift(self, current_epoch: int) -> dict:
        """Check for topic drift and determine if preference refresh is needed."""
        if self.drift_monitor is None:
            self.initialize_drift_monitoring()
        
        if self.drift_monitor is not None:
            beta = self.get_beta()
            drift_result = self.drift_monitor.check_drift(beta, current_epoch)
            
            if drift_result["should_refresh"]:
                self.preference_refresh_needed = True
                print(f"🚨 TOPIC DRIFT DETECTED! Will refresh preference dataset.")
                print(f"   Mean Jaccard: {drift_result['mean_jaccard']:.3f}")
                print(f"   Drifted topics: {drift_result['num_drifted']}/{self.num_topics}")
            
            return drift_result
        
        return {"drift_detected": False, "should_refresh": False}
    
    def refresh_preference_dataset(self):
        """Refresh preference dataset when topics drift too much."""
        if not self.preference_refresh_needed:
            return False
            
        try:
            print("🔄 Refreshing preference dataset due to topic drift...")
            
            # 1. Save current beta as new top words
            beta = self.get_beta()
            top_words, top_word_indices = self.export_top_words_for_preference()
            
            # 2. Create new preference dataset using LLM
            preference_creator = PreferenceDatasetCreator(self.current_run_dir)
            preference_creator.create()
            
            # 3. Clear cached preference data to force reload
            self.preference_cache = None
            self.preference_dataset = None
            
            # 4. Update drift monitor reference
            if self.drift_monitor is not None:
                self.drift_monitor.update_reference(beta)
            
            # 5. Update reference beta
            self.beta_ref = beta.clone().detach()
            self.beta_ref.requires_grad = False
            
            self.preference_refresh_needed = False
            print("✅ Preference dataset refreshed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to refresh preference dataset: {e}")
            self.preference_refresh_needed = False  # Reset to avoid infinite loops
            return False
    
    def export_top_words_for_preference(self, num_top_words: int = 25):
        """Export top words in format needed for preference dataset creation."""
        beta = self.get_beta().detach().cpu().numpy()
        
        # Get top word indices for each topic
        top_word_indices = []
        for k in range(self.num_topics):
            top_indices = np.argsort(beta[k])[::-1][:num_top_words]
            top_word_indices.append(top_indices.tolist())
        
        # Save in JSONL format for preference dataset creation
        top_words_path = os.path.join(self.current_run_dir, f'top_words_{num_top_words}.jsonl')
        with open(top_words_path, 'w') as f:
            for k, indices in enumerate(top_word_indices):
                top_words_with_indices = []
                for idx in indices:
                    word = self.vocab[idx] if self.vocab else f"word_{idx}"
                    top_words_with_indices.append({word: idx})
                
                topic_data = {
                    'k': k,
                    'top_words': top_words_with_indices
                }
                f.write(json.dumps(topic_data) + '\n')
        
        return top_word_indices, top_words_path
    
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
        # Short-circuit if DPO is disabled
        if getattr(self, 'lambda_dpo', 0.0) <= 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if self.preference_dataset is None:
            self.load_preference_dataset()
        
        beta = self.get_beta()
        
        # Enhanced preference pairing với quality control
        if self.preference_cache is None:
            self.preference_cache = []
            for line in self.preference_dataset:
                data = json.loads(line)
                k = data['k']
                w_plus_indices = data['w_plus_indices'] 
                w_minus_indices = data['w_minus_indices']
                
                # AGGRESSIVE pairing strategy cho stronger signal
                min_pairs = min(len(w_plus_indices), len(w_minus_indices))
                max_pairs_per_topic = min(40, min_pairs * 5)  # TĂNG từ 20 → 40 pairs
                
                # Strategy 1: Best vs Worst pairing - TĂNG số lượng
                for i in range(min(10, min_pairs)):  # TĂNG từ 5 → 10
                    w_plus_idx = w_plus_indices[i]  # Top words
                    w_minus_idx = w_minus_indices[-(i+1)]  # Bottom words
                    self.preference_cache.append((k, w_plus_idx, w_minus_idx))
                
                # Strategy 2: Random balanced pairing cho diversity
                for i in range(10, max_pairs_per_topic):  # Tăng range
                    w_plus_idx = w_plus_indices[i % len(w_plus_indices)]
                    w_minus_idx = w_minus_indices[i % len(w_minus_indices)]
                    self.preference_cache.append((k, w_plus_idx, w_minus_idx))
        
        # AGGRESSIVE batch processing để force learning
        batch_size = min(768, len(self.preference_cache))  # TĂNG từ 512 → 768 cho more data per batch
        if len(self.preference_cache) > batch_size:
            # Weighted sampling: ưu tiên pairs với high confidence
            import random
            batch_indices = random.sample(range(len(self.preference_cache)), batch_size)
            batch_preferences = [self.preference_cache[i] for i in batch_indices]
        else:
            batch_preferences = self.preference_cache
            
        chosen_logps = []
        rejected_logps = []
        ref_chosen_logps = []
        ref_rejected_logps = []
        
        # AGGRESSIVE DPO mode với stronger temperature
        temperature = 0.2  # GIẢM từ 0.3 → 0.2 cho sharper, stronger signal
        
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
        
        # Enhanced DPO loss với adaptive weighting
        pi_logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios
        
        # Moderate clamping để preserve signal strength
        logits = torch.clamp(logits, -5.0, 5.0)  # Giảm từ [-10,10] → [-5,5]
        
        if self.use_ipo:
            # IPO loss cho training ổn định hơn
            losses = (logits - 1/(2 * self.lambda_dpo)) ** 2
        else:
            # Enhanced DPO với confidence weighting
            sigmoid_logits = F.logsigmoid(self.lambda_dpo * logits)
            neg_sigmoid_logits = F.logsigmoid(-self.lambda_dpo * logits)
            
            # Confidence-weighted loss: stronger signal cho high-confidence pairs
            confidence_weights = torch.abs(logits).detach()  # Higher weight cho clear preferences
            confidence_weights = torch.clamp(confidence_weights, 0.5, 2.0)  # Normalize weights
            
            losses = (-sigmoid_logits * (1 - self.label_smoothing) - 
                     neg_sigmoid_logits * self.label_smoothing) * confidence_weights
        
        # Compute rewards cho monitoring
        chosen_rewards = self.lambda_dpo * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.lambda_dpo * (rejected_logps - ref_rejected_logps).detach()
        
        # Enhanced metrics tracking
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        self.reward_accuracies = reward_accuracies.cpu().numpy().tolist()
        self.reward_margins = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
        
        # AGGRESSIVE adaptive scaling với stricter thresholds
        avg_acc = reward_accuracies.mean().item()
        if avg_acc < 0.5:  # Very poor performance - STRICTER threshold
            losses = losses * 0.5  # Reduce significantly  
        elif avg_acc > 0.7:  # Good performance - LOWER threshold
            losses = losses * 1.5  # AMPLIFY more aggressively
        elif avg_acc > 0.9:  # Excellent performance
            losses = losses * 2.0  # MAXIMUM amplification
        
        return losses.mean()

    def get_loss_regularization(self):
        beta = self.get_beta()
        
        # Enhanced regularization specifically cho TC_15 improvement
        # 1. Moderate L2 regularization - không quá strict
        l2_reg = torch.mean((beta - self.beta_ref) ** 2) * 0.5  # Giảm weight
        
        # 2. Topic coherence regularization - encourage coherent topics
        # Tăng probability mass cho top words của mỗi topic
        beta_sorted, _ = torch.sort(beta, dim=1, descending=True)
        top_k_mass = torch.sum(beta_sorted[:, :15], dim=1)  # Top 15 words mass
        coherence_reg = -torch.mean(top_k_mass)  # Maximize top-k mass
        
        # 3. Controlled diversity - tránh topics quá giống nhau
        beta_norm = F.normalize(beta, dim=1, p=2)
        cosine_sim = torch.matmul(beta_norm, beta_norm.t())
        # Chỉ penalize similarity quá cao (> 0.7)
        high_sim_mask = (cosine_sim > 0.7).float()
        diversity_reg = torch.mean(torch.triu(cosine_sim * high_sim_mask, diagonal=1))
        
        # 4. Sparsity regularization - encourage focused topics
        # Entropy-based sparsity: lower entropy = more focused
        entropy_reg = torch.mean(torch.sum(-beta * torch.log(beta + 1e-8), dim=1))
        
        # 5. Reference similarity maintenance - đừng drift quá xa
        cos_sim_ref = F.cosine_similarity(beta.view(-1), self.beta_ref.view(-1), dim=0)
        ref_sim_reg = -(cos_sim_ref - 0.8) ** 2  # Encourage similarity around 0.8
        
        # Weighted combination với focus on coherence improvement
        total_reg = (0.3 * l2_reg + 
                    0.4 * coherence_reg + 
                    0.1 * diversity_reg + 
                    0.1 * entropy_reg + 
                    0.1 * ref_sim_reg)
        
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
            
            # AGGRESSIVE loss weighting cho maximum DPO impact
            with torch.no_grad():
                loss_magnitudes = {
                    'TM': loss_TM.item(),
                    'ECR': loss_ECR.item(), 
                    'DPO': loss_DPO.item(),
                    'REG': loss_regularization.item()
                }
                
                # MAXIMUM DPO scaling strategy
                base_loss = loss_magnitudes['TM'] + loss_magnitudes['ECR']
                if base_loss > 0:
                    # Allow DPO to scale up to 3x lambda_dpo cho maximum impact
                    dpo_scale = min(self.lambda_dpo * 3.0, base_loss / max(loss_magnitudes['DPO'], 1e-8)) if self.lambda_dpo > 0 else 0.0
                    dpo_scale = max(dpo_scale, self.lambda_dpo * 0.8) if self.lambda_dpo > 0 else 0.0
                    
                    reg_scale = min(self.lambda_reg * 2.0, base_loss / max(loss_magnitudes['REG'], 1e-8))
                else:
                    dpo_scale = self.lambda_dpo * 1.5 if self.lambda_dpo > 0 else 0.0
                    reg_scale = self.lambda_reg * 1.5
            
            # Final loss với AGGRESSIVE DPO weighting
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
import torch
import numpy as np
from typing import List, Dict, Tuple
import logging


class TopicDriftMonitor:
    """
    Monitor topic drift during fine-tuning using Jaccard overlap coefficient.
    If topics drift too much, trigger preference dataset refresh.
    """
    
    def __init__(self, vocab: List[str], num_topics: int = 50, top_k: int = 25, 
                 jaccard_threshold: float = 0.8, check_interval: int = 10):
        """
        Args:
            vocab: Vocabulary list
            num_topics: Number of topics
            top_k: Number of top words to compare for Jaccard
            jaccard_threshold: Threshold below which we consider topic drifted
            check_interval: Check drift every N epochs
        """
        self.vocab = vocab
        self.num_topics = num_topics
        self.top_k = top_k
        self.jaccard_threshold = jaccard_threshold
        self.check_interval = check_interval
        
        # Store reference top words from the beginning of fine-tuning
        self.reference_top_words = None
        self.last_check_epoch = 0
        self.drift_detected = False
        
        self.logger = logging.getLogger('drift_monitor')
    
    def get_top_words_indices(self, beta: torch.Tensor, k: int = None) -> List[List[int]]:
        """
        Get top-k word indices for each topic.
        
        Args:
            beta: Topic-word distribution matrix [num_topics, vocab_size]
            k: Number of top words (default: self.top_k)
            
        Returns:
            List of top-k word indices for each topic
        """
        if k is None:
            k = self.top_k
            
        # Get top-k indices for each topic
        _, top_indices = torch.topk(beta, k, dim=1)
        return [indices.cpu().numpy().tolist() for indices in top_indices]
    
    def compute_jaccard_similarity(self, set_a: set, set_b: set) -> float:
        """
        Compute Jaccard similarity between two sets.
        
        Args:
            set_a, set_b: Two sets to compare
            
        Returns:
            Jaccard similarity coefficient [0, 1]
        """
        if len(set_a) == 0 and len(set_b) == 0:
            return 1.0
        
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        
        return intersection / union if union > 0 else 0.0
    
    def compute_topic_jaccard_scores(self, current_top_words: List[List[int]], 
                                   reference_top_words: List[List[int]]) -> List[float]:
        """
        Compute Jaccard scores for all topics.
        
        Args:
            current_top_words: Current top words for each topic
            reference_top_words: Reference top words for each topic
            
        Returns:
            List of Jaccard scores for each topic
        """
        jaccard_scores = []
        
        for topic_idx in range(len(current_top_words)):
            current_set = set(current_top_words[topic_idx])
            reference_set = set(reference_top_words[topic_idx])
            
            jaccard = self.compute_jaccard_similarity(current_set, reference_set)
            jaccard_scores.append(jaccard)
        
        return jaccard_scores
    
    def initialize_reference(self, beta: torch.Tensor):
        """
        Initialize reference top words from beta at start of fine-tuning.
        
        Args:
            beta: Initial topic-word distribution matrix
        """
        self.reference_top_words = self.get_top_words_indices(beta)
        self.logger.info(f"Initialized reference top-{self.top_k} words for {self.num_topics} topics")
    
    def check_drift(self, beta: torch.Tensor, current_epoch: int) -> Dict[str, any]:
        """
        Check if topics have drifted based on Jaccard overlap.
        
        Args:
            beta: Current topic-word distribution matrix
            current_epoch: Current epoch number
            
        Returns:
            Dict with drift analysis results
        """
        # Initialize reference if not done yet
        if self.reference_top_words is None:
            self.initialize_reference(beta)
            return {"drift_detected": False, "should_refresh": False}
        
        # Only check at specified intervals
        if current_epoch - self.last_check_epoch < self.check_interval:
            return {"drift_detected": False, "should_refresh": False}
        
        self.last_check_epoch = current_epoch
        
        # Get current top words
        current_top_words = self.get_top_words_indices(beta)
        
        # Compute Jaccard scores
        jaccard_scores = self.compute_topic_jaccard_scores(current_top_words, self.reference_top_words)
        
        # Analyze drift
        mean_jaccard = np.mean(jaccard_scores)
        min_jaccard = np.min(jaccard_scores)
        drifted_topics = [i for i, score in enumerate(jaccard_scores) if score < self.jaccard_threshold]
        
        # Determine if refresh is needed
        should_refresh = (mean_jaccard < self.jaccard_threshold or 
                         len(drifted_topics) > self.num_topics * 0.3)  # >30% topics drifted
        
        if should_refresh:
            self.drift_detected = True
        
        result = {
            "drift_detected": should_refresh,
            "should_refresh": should_refresh,
            "mean_jaccard": mean_jaccard,
            "min_jaccard": min_jaccard,
            "jaccard_scores": jaccard_scores,
            "drifted_topics": drifted_topics,
            "num_drifted": len(drifted_topics),
            "drift_ratio": len(drifted_topics) / self.num_topics
        }
        
        # Log results
        if should_refresh:
            self.logger.warning(
                f"TOPIC DRIFT DETECTED at epoch {current_epoch}! "
                f"Mean Jaccard: {mean_jaccard:.3f}, "
                f"Drifted topics: {len(drifted_topics)}/{self.num_topics} ({len(drifted_topics)/self.num_topics:.1%})"
            )
            
            # Log worst drifted topics
            worst_topics = sorted(enumerate(jaccard_scores), key=lambda x: x[1])[:5]
            for topic_idx, score in worst_topics:
                current_words = [self.vocab[idx] for idx in current_top_words[topic_idx][:10]]
                reference_words = [self.vocab[idx] for idx in self.reference_top_words[topic_idx][:10]]
                self.logger.warning(
                    f"Topic {topic_idx} (Jaccard: {score:.3f}): "
                    f"Current: {current_words[:5]} | Reference: {reference_words[:5]}"
                )
        else:
            self.logger.info(
                f"Topic stability check at epoch {current_epoch}: "
                f"Mean Jaccard: {mean_jaccard:.3f}, "
                f"Drifted: {len(drifted_topics)}/{self.num_topics}"
            )
        
        return result
    
    def update_reference(self, beta: torch.Tensor):
        """
        Update reference top words after preference dataset refresh.
        
        Args:
            beta: New topic-word distribution matrix
        """
        self.reference_top_words = self.get_top_words_indices(beta)
        self.drift_detected = False
        self.logger.info(f"Updated reference top-{self.top_k} words after preference refresh")

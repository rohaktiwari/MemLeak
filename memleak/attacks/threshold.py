import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from .base import MemberInferenceAttack

class ThresholdAttack(MemberInferenceAttack):
    def __init__(self, metric='confidence', config=None):
        super().__init__(config)
        self.metric = metric # 'confidence', 'loss', 'entropy'
        self.threshold = None

    def _compute_metric(self, outputs, targets=None, criterion=None):
        probs = torch.softmax(outputs, dim=1)
        
        if self.metric == 'confidence':
            # Max probability
            scores, _ = probs.max(dim=1)
            return scores.cpu().numpy()
            
        elif self.metric == 'loss':
            if targets is None or criterion is None:
                raise ValueError("Loss metric requires targets and criterion")
            
            losses = []
            criterion_reduce_none = torch.nn.CrossEntropyLoss(reduction='none')
            with torch.no_grad():
                l = criterion_reduce_none(outputs, targets)
                losses = l.cpu().numpy()
            
            # For MIA, lower loss usually means member. 
            # To make "higher score = more likely member", we can negate loss.
            return -losses 
            
        elif self.metric == 'entropy':
            # -sum(p * log(p))
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            # Lower entropy -> member. Score = -entropy
            return -entropy.cpu().numpy()
            
        else:
            raise ValueError(f"Unknown metric {self.metric}")

    def train_attack_model(self, shadow_member_outputs, shadow_non_member_outputs, targets_member=None, targets_non_member=None, criterion=None):
        # Find optimal threshold using shadow data
        
        # Combine data
        scores_member = self._compute_metric(shadow_member_outputs, targets_member, criterion)
        scores_non_member = self._compute_metric(shadow_non_member_outputs, targets_non_member, criterion)
        
        y_scores = np.concatenate([scores_member, scores_non_member])
        y_true = np.concatenate([np.ones(len(scores_member)), np.zeros(len(scores_non_member))])
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # Pick threshold that maximizes accuracy or Youden's J statistic
        J = tpr - fpr
        ix = np.argmax(J)
        self.threshold = thresholds[ix]
        
        return {'threshold': self.threshold, 'auc': auc(fpr, tpr)}

    def predict(self, outputs, targets=None, criterion=None):
        scores = self._compute_metric(outputs, targets, criterion)
        if self.threshold is None:
             # If no threshold trained, return raw scores (higher is more member-like)
             return scores
             
        return (scores >= self.threshold).astype(int)

    def run(self, model, member_loader, non_member_loader, device='cuda'):
        # This method assumes we just want to execute the attack given a model and data
        # Usually requires `train_attack_model` to have been called if we want hard predictions.
        # But we can also just return the ROC curve Area.
        pass

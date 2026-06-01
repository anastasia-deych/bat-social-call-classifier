import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    make_scorer,
    roc_auc_score,
    hamming_loss,
    accuracy_score,
    f1_score,
)
import warnings
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import torch.nn.functional as F
#from transformers import PipelineedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
warnings.filterwarnings('ignore')

from models.abmil_model import  ABMILSklearnWrapper,predict_abmil

#class HierarchicalMultiTaskLoss(nn.Module):
#    def __init__(self, lambda_penalty=2.0):
#        super().__init__()
#        self.lambda_penalty = lambda_penalty
#        self.bce_loss = nn.BCEWithLogitsLoss()
#
#    def forward(self, logits, targets):
#        """
#        logits/targets shape: (6,) -> [Echo, Social, A, B, C, D]
#        """
#        # Split macro vs micro
#        macro_logits, micro_logits = logits[:2], logits[2:]
#        y_macro, y_micro = targets[:2], targets[2:]
#        
#        # Standard BCE Losses
#        loss_macro = self.bce_loss(macro_logits, y_macro)
#        loss_micro = self.bce_loss(micro_logits, y_micro)
#        
#        # Convert to probabilities for penalty logic
#        p_macro = torch.sigmoid(macro_logits)
#        p_micro = torch.sigmoid(micro_logits)
#        
#        # Index 1 of macro is 'Social_Call'
#        p_social = p_macro[1] 
#        
#        # Penalty triggers if micro prob exceeds macro social prob
#        violations = F.relu(p_micro - p_social)
#        penalty_term = torch.sum(violations ** 2)
#        
#        total_loss = loss_macro + loss_micro + (self.lambda_penalty * penalty_term)
#        return total_loss

#class HierarchicalMultiTaskLoss(nn.Module):
#    def __init__(self, lambda_penalty=1.0):
#        super().__init__()
#        self.lambda_penalty = lambda_penalty
#        # Use basic BCELoss because we will pass explicit probabilities manually
#        self.bce_loss = nn.BCELoss()
#
#    def forward(self, logits, targets):
#        """
#        logits/targets shape: (batch_size, 6) -> [Echo, Social, A, B, C, D]
#        """
#        # 1. Convert everything to probability space safely before computing loss
#        probs = torch.sigmoid(logits)
#        
#        # Avoid numerical instability (log(0)) by clamping probabilities slightly
#        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
#        if probs.dim() == 1:
#            probs = probs.unsqueeze(0)
#            targets = targets.unsqueeze(0)
#
#        # 2. Compute true binary cross-entropy on stable probabilities
#        loss_macro = self.bce_loss(probs[:, :2], targets[:, :2])
#        loss_micro = self.bce_loss(probs[:, 2:], targets[:, 2:])
#        
#        # 3. Targeted Hierarchy Penalty
#        p_social = probs[:, 1:2]   # Shape: (batch_size, 1) - Macro Social Call Prob
#        p_types = probs[:, 2:]     # Shape: (batch_size, 4) - Micro Types A, B, C, D
#        
#        # Hinge Loss: Only penalize if a specific Type probability exceeds the parent Social probability
#        # F.relu ensures that if p_types <= p_social, the violation is exactly 0.0
#        violations = F.relu(p_types - p_social)
#        
#        # Mean across the batch to keep it balanced with standard BCE loss values
#        penalty_term = torch.mean(violations ** 2)
#        
#        # 4. Combine smoothly
#        total_loss = loss_macro + loss_micro + (self.lambda_penalty * penalty_term)
#        return total_loss

class HierarchicalMultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.bce(logits, targets)



# ============================================================================
# HIERARCHICAL CLASSIFIER (Refactored to Unified Multi-Task)
# ============================================================================

class HierarchicalABMIL:
    """
    Multi-task hierarchical ABMIL classifier.
    Trains ONE unified network with a penalty bridging the two tasks.
    """
    
    def __init__(self, device='cpu', verbose=True):
        self.device = device
        self.verbose = verbose
        self.mt_results = None
        
    def prepare_hierarchical_labels(self, y_original):
        """
        Input: [Echo, TypeA, TypeB, TypeC, TypeD]
        Output: [Echo, Social_Call, TypeA, TypeB, TypeC, TypeD] (Shape: N x 6)
        """
        y = np.array(y_original)
        
        # Extract Social_Call flag
        social_call = np.max(y[:, 1:], axis=1, keepdims=True)
        
        # Combine into unified 6-label target
        y_combined = np.hstack([
            y[:, 0:1],       # Echo
            social_call,     # Social Call Macro
            y[:, 1:]         # Micro Types A-D
        ])
        
        return y_combined
    
    def fit(self, X_bags, y_original, mt_params=None,
            n_splits_outer=5, n_splits_inner=3, 
            n_iter_search=4, random_state=42):
        
        # 1. Prepare unified 6-label array
        y_unified = self.prepare_hierarchical_labels(y_original)
        
        if self.verbose:
            print("="*70)
            print("TRAINING UNIFIED MULTI-TASK HIERARCHY (6 Labels)")
            print("="*70)
        
        # Default hyperparameter search space (Now includes lambda_penalty)
        if mt_params is None:
            mt_params = {
                'hidden_dim': [64, 128, 256],
                'attention_dim': [32, 64, 128],
                'dropout': [0.1, 0.3, 0.5],
                'learning_rate': loguniform(1e-4, 5e-3),
                'weight_decay': loguniform(1e-6, 1e-2),
                'lambda_penalty': [0.5, 1.0, 2.5, 5.0] # Exposed for CV!
            }
        
        inner_cv = MultilabelStratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=random_state)
        outer_cv = MultilabelStratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=random_state)
        
        scorer = make_scorer(average_precision_score, average='macro', response_method='predict_proba')
        
        oof_y_true = []
        oof_y_pred = []
        outer_scores = []
        best_models_per_fold = []
        
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_bags, y_unified)):
            if self.verbose:
                print(f"\nFold {fold+1}/{n_splits_outer}")
            
            X_bags_train = [X_bags[i] for i in train_idx]
            X_bags_test = [X_bags[i] for i in test_idx]
            y_train, y_test = y_unified[train_idx], y_unified[test_idx]
            
            clf = RandomizedSearchCV(
                estimator=ABMILSklearnWrapper(
                    n_labels=6, n_epochs=40, batch_size=4,hierarchal = True, device=self.device
                ),
                param_distributions=mt_params,
                n_iter=n_iter_search,
                cv=inner_cv,
                scoring=scorer,
                refit=True,
                n_jobs=1,
                random_state=random_state + fold,
                verbose=3 if self.verbose == True else 0
            )
            
            clf.fit(X_bags_train, y_train)
            best_models_per_fold.append(clf.best_estimator_)
            
            #y_pred_proba = clf.predict_proba(X_bags_test)

            # 2. Get raw 6-label predictions from the trained estimator
            # Using clf.best_estimator_.model_ directly ensures we control the mapping
            y_pred_6, _ = predict_abmil(
                clf.best_estimator_.model_, X_bags_test, clf.best_estimator_.scaler_,
                device=self.device, hierarchal=True # Ensures we get clean [0,1] probabilities
            )
            
            # 3. Apply Soft-Gating Transformation right here for proper evaluation
            y_pred_stage1 = y_pred_6[:, :2]     # [Echo, Social_Call]
            y_pred_stage2_raw = y_pred_6[:, 2:]  # [Type_A, B, C, D]
            
            # Soft gating multiplier: P(Type) = P(Social) * P(Type | Social)
            p_social = y_pred_stage1[:, [1]] 
            #y_pred_stage2_gated = y_pred_stage2_raw * p_social
            
            # Recombine into your standard 5-label array: [Echo, Type A, Type B, Type C, Type D]
            y_pred_5_gated = np.column_stack([
                y_pred_stage1[:, 0], 
                y_pred_stage2_raw
            ])
            
            # 4. Map y_test back to original 5 labels for an authentic performance score
            y_test_5 = np.column_stack([y_test[:, 0], y_test[:, 2:]])
            
            # Compute real 5-label Macro Average Precision
            fold_ap = average_precision_score(y_test_5, y_pred_5_gated, average='macro')
            outer_scores.append(fold_ap)
            
            if self.verbose:
                print(f"  Best params: {clf.best_params_}")
                print(f"  Gated Test AP (Original 5 Labels): {fold_ap:.4f}")
            
            # Append genuine 5-label spaces to out-of-fold arrays
            oof_y_true.append(y_test_5)
            oof_y_pred.append(y_pred_5_gated)
            
            """
            if isinstance(y_pred_proba, list):
                y_pred_proba = np.column_stack([prob[:, 1] for prob in y_pred_proba])
            elif isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 3:
                y_pred_proba = y_pred_proba[:, :, 1].T
            
            fold_ap = average_precision_score(y_test, y_pred_proba, average='macro')
            outer_scores.append(fold_ap)
            
            if self.verbose:
                print(f"  Best params: {clf.best_params_}")
                print(f"  Test AP (6 Labels): {fold_ap:.4f}")
            
            oof_y_true.append(y_test)
            oof_y_pred.append(y_pred_proba)
            """
        
        oof_y_true = np.concatenate(oof_y_true, axis=0)
        oof_y_pred = np.concatenate(oof_y_pred, axis=0)
        
        overall_ap = average_precision_score(oof_y_true, oof_y_pred, average='macro')
        overall_auc = roc_auc_score(oof_y_true, oof_y_pred, average='macro')
        
        if self.verbose:
            print(f"\nMulti-Task CV Results:")
            print(f"  OOF Overall AP: {overall_ap:.4f}")
            print(f"  OOF Overall AUC: {overall_auc:.4f}")
            
        self.mt_results = {
            'best_models': best_models_per_fold,
            'oof_y_true': oof_y_true,
            'oof_y_pred': oof_y_pred,
            'overall_ap': overall_ap,
            'overall_auc': overall_auc
        }
        
        return self
    
    def predict(self, X_bags):
        """
        Output filtering mimicking the 2-stage logic.
        """
        best_model = self.mt_results['best_models'][0]
        
        # Get raw 6-label predictions
        y_pred_6, _ = predict_abmil(
            best_model.model_, X_bags, best_model.scaler_,
            device=self.device, return_attention=False,hierarchal = True
        )
        
        # Split logic
        y_pred_stage1 = y_pred_6[:, :2]    # [Echo, Social_Call]
        y_pred_stage2_raw = y_pred_6[:, 2:] # [Type_A, B, C, D]
        
        # Filter: Zero out Types A-D if Social_Call (index 1) is < 0.5
        y_pred_stage2_filtered = y_pred_stage2_raw.copy()
        #social_call_detected = y_pred_stage1[:, 1] > 0.5
        #y_pred_stage2_filtered[~social_call_detected] = 0
        
        # Original format recombination: [Echo, A, B, C, D]
        y_combined = np.column_stack([
            y_pred_stage1[:, 0], 
            y_pred_stage2_filtered
        ])
        
        return y_pred_stage1, y_pred_stage2_raw, y_pred_stage2_filtered, y_combined
"""
Multiple Instance Learning (MIL) for Multi-Label Bioacoustic Classification.

Implements:
- ABMIL (Attention-Based MIL): Uses attention mechanism to weight instance importance
- Multi-label setup: Each recording has multiple labels
- Window-level features: Uses un-pooled window embeddings instead of mean-pooled

Key concept: Instead of mean-pooling all windows, ABMIL learns which windows
are most important for predicting each label.
"""

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
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
#from transformers import PipelineedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
warnings.filterwarnings('ignore')


# ============================================================================
# DATASET CLASS
# ============================================================================

class MILDataset(Dataset):
    """Dataset for Multiple Instance Learning."""
    
    def __init__(self, X_bags, y_labels, scaler=None, fit_scaler=False):
        """
        Parameters
        ----------
        X_bags : list of (n_windows, n_features) arrays
            One bag (recording) per element. IMPORTANT: bags can have different lengths!
        y_labels : (n_bags, n_labels) array
            Multi-label targets
        scaler : StandardScaler or None
        fit_scaler : bool
            If True, fit scaler on this data
        """
        self.X_bags = X_bags
        self.y_labels = torch.FloatTensor(y_labels)
        self.scaler = scaler
        
        if fit_scaler and scaler is not None:
            # Fit on all instances concatenated
            all_instances = np.vstack(X_bags)
            scaler.fit(all_instances)
        
        # Scale all bags
        self.X_bags_scaled = []
        for bag in X_bags:
            if scaler is not None:
                bag_scaled = scaler.transform(bag)
            else:
                bag_scaled = bag
            self.X_bags_scaled.append(torch.FloatTensor(bag_scaled))
    
    def __len__(self):
        return len(self.X_bags_scaled)
    
    def __getitem__(self, idx):
        return self.X_bags_scaled[idx], self.y_labels[idx]


def collate_mil(batch):
    """
    Custom collate function for MIL batches with variable-length bags.
    
    PyTorch's default collate tries to stack tensors, which fails when
    bags have different numbers of instances.
    
    Parameters
    ----------
    batch : list of tuples
        Each tuple is (bag_tensor, label_tensor) where bag_tensor
        can have different length
    
    Returns
    -------
    bags : list of tensors
        Each element is a (n_instances_i, n_features) tensor
    labels : (batch_size, n_labels) tensor
        Stacked labels
    """
    bags = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return bags, labels


# ============================================================================
# ABMIL MODEL
# ============================================================================

class ABMIL(nn.Module):
    """
    Attention-Based Multiple Instance Learning.
    
    Architecture:
    1. Feature extraction (optional): Maps instance features through FC layer
    2. Attention mechanism: Learns which instances are important
    3. Aggregation: Weighted sum using attention weights
    4. Classification: Multi-label head (sigmoid)
    
    For multi-label: Trains one attention module per label (label-specific attention)
    """
    
    def __init__(self, n_features, n_labels, hidden_dim=256, dropout=0.2, 
                 attention_dim=128, label_specific_attention=True):
        """
        Parameters
        ----------
        n_features : int
            Dimension of instance features
        n_labels : int
            Number of labels
        hidden_dim : int
            Hidden dimension for feature processing
        dropout : float
            Dropout probability
        attention_dim : int
            Dimension of attention mechanism
        label_specific_attention : bool
            If True, use separate attention for each label (recommended for multi-label)
        """
        super().__init__()
        
        self.n_features = n_features
        self.n_labels = n_labels
        self.label_specific_attention = label_specific_attention
        
        # Feature processing
        self.feature_fc = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        if label_specific_attention:
            # Separate attention for each label
            self.attention_modules = nn.ModuleList([
                self._build_attention(hidden_dim, attention_dim)
                for _ in range(n_labels)
            ])
        else:
            # Shared attention across all labels
            self.attention = self._build_attention(hidden_dim, attention_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, n_labels),
            nn.Sigmoid()
        )
    
    @staticmethod
    def _build_attention(hidden_dim, attention_dim):
        """Build attention module."""
        return nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
    
    def forward(self, x_bag, return_attention=False):
        """
        Forward pass.
        
        Parameters
        ----------
        x_bag : (n_instances, n_features)
            Instances from one bag
        return_attention : bool
            If True, return attention weights
        
        Returns
        -------
        logits : (n_labels,)
            Prediction logits
        attention_weights : list of (n_instances,) or None
            Attention weights per label
        """
        # Feature processing
        H = self.feature_fc(x_bag)  # (n_instances, hidden_dim)
        
        if self.label_specific_attention:
            # Compute attention per label
            logits_list = []
            attention_weights_list = []
            
            for label_idx in range(self.n_labels):
                # Attention for this label
                A = self.attention_modules[label_idx](H)  # (n_instances, 1)
                A = torch.softmax(A, dim=0)  # (n_instances, 1)
                A_squeezed = A.squeeze(1)  # (n_instances,)
                
                # Weighted aggregation
                M = torch.sum(A * H, dim=0, keepdim=True)  # (1, hidden_dim)
                
                # Classification
                logit = self.classifier(M)[0, label_idx]  
                logits_list.append(logit)
                attention_weights_list.append(A_squeezed)
            
            logits = torch.stack(logits_list)
            attention_weights = attention_weights_list if return_attention else None
        else:
            # Shared attention across labels
            A = self.attention(H)  # (n_instances, 1)
            A = torch.softmax(A, dim=0)  # (n_instances, 1)
            
            # Weighted aggregation
            M = torch.sum(A * H, dim=0, keepdim=True)  # (1, hidden_dim)
            
            # Classification
            logits = self.classifier(M).squeeze(0)  # (n_labels,)
            attention_weights = A.squeeze(1) if return_attention else None
        
        return logits, attention_weights


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_abmil(
    X_bags_train,
    y_train,
    X_bags_val=None,
    y_val=None,
    n_labels=5,
    n_epochs=50,
    batch_size=16,
    learning_rate=1e-3,
    weight_decay=1e-5,
    hidden_dim=256,
    attention_dim=128,
    dropout=0.2,
    label_specific_attention=True,
    device='cpu',
    verbose=True,
):
    """
    Train ABMIL model.
    
    Parameters
    ----------
    X_bags_train : list of arrays
        Training bags (recordings)
    y_train : (n_bags, n_labels) array
        Training labels
    X_bags_val : list of arrays or None
        Validation bags
    y_val : array or None
        Validation labels
    n_labels : int
    n_epochs : int
    batch_size : int
    learning_rate : float
    weight_decay : float
    hidden_dim : int
    attention_dim : int
    dropout : float
    label_specific_attention : bool
    device : str
        'cpu' or 'cuda'
    verbose : bool
    
    Returns
    -------
    model : ABMIL
        Trained model
    history : dict
        Training history
    scaler : StandardScaler
        Feature scaler
    """
    
    # Setup device
    device = torch.device(device)
    
    # Scaler
    scaler = StandardScaler()
    
    # Create datasets
    train_dataset = MILDataset(X_bags_train, y_train, scaler=scaler, fit_scaler=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              collate_fn=collate_mil)
    
    if X_bags_val is not None:
        val_dataset = MILDataset(X_bags_val, y_val, scaler=scaler, fit_scaler=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               collate_fn=collate_mil)
    else:
        val_loader = None
    
    # Get feature dimension
    n_features = train_dataset.X_bags_scaled[0].shape[1]
    
    # Model
    model = ABMIL(
        n_features=n_features,
        n_labels=n_labels,
        hidden_dim=hidden_dim,
        attention_dim=attention_dim,
        dropout=dropout,
        label_specific_attention=label_specific_attention,
    ).to(device)
    
    # Loss & optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5 #, verbose=verbose
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_ap': [],
        'val_auc': [],
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            # X_batch is now a list of tensors (variable length bags)
            # y_batch is a (batch_size, n_labels) tensor
            
            optimizer.zero_grad()
            
            batch_loss = 0.0
            for bag_idx in range(len(X_batch)):
                X_bag = X_batch[bag_idx].to(device)  # (n_instances, n_features)
                y_bag = y_batch[bag_idx].to(device)   # (n_labels,)
                
                logits, _ = model(X_bag)
                loss = criterion(logits, y_bag)
                batch_loss += loss
            
            batch_loss = batch_loss / len(X_batch)
            batch_loss.backward()
            optimizer.step()
            
            train_loss += batch_loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            all_y_true = []
            all_y_pred = []
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    # X_batch is a list of tensors (variable length)
                    # y_batch is (batch_size, n_labels)
                    
                    batch_loss = 0.0
                    
                    for bag_idx in range(len(X_batch)):
                        X_bag = X_batch[bag_idx].to(device)
                        y_bag = y_batch[bag_idx].to(device)
                        
                        logits, _ = model(X_bag)
                        loss = criterion(logits, y_bag)
                        batch_loss += loss
                        
                        all_y_true.append(y_bag.cpu().numpy())
                        all_y_pred.append(logits.detach().cpu().numpy())
                    
                    batch_loss = batch_loss / len(X_batch)
                    val_loss += batch_loss.item()
            
            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)
            
            # Metrics
            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)
            
            val_ap = average_precision_score(all_y_true, all_y_pred, average='macro')
            val_auc = roc_auc_score(all_y_true, all_y_pred, average='macro')
            
            history['val_ap'].append(val_ap)
            history['val_auc'].append(val_auc)
            
            # LR scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f} | "
                     f"Val Loss: {val_loss:.4f} | Val AP: {val_ap:.4f} | Val AUC: {val_auc:.4f}")
            
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
    
    return model, history, scaler


def predict_abmil(model, X_bags, scaler, device='cpu'):
    """
    Get predictions from trained ABMIL model.
    
    Parameters
    ----------
    model : ABMIL
    X_bags : list of arrays
    scaler : StandardScaler
    device : str
    
    Returns
    -------
    y_pred : (n_bags, n_labels) array
        Predictions
    attention_weights : dict
        Attention weights per bag and label
    """
    device = torch.device(device)
    model.to(device)
    model.eval()
    
    predictions = []
    attention_dict = {}
    
    with torch.no_grad():
        for bag_idx, X_bag in enumerate(X_bags):
            # Scale
            X_bag_scaled = scaler.transform(X_bag)
            X_bag_tensor = torch.FloatTensor(X_bag_scaled).to(device)
            
            # Predict
            logits, attention_weights = model(X_bag_tensor, return_attention=True)
            y_pred = logits.detach().cpu().numpy()
            predictions.append(y_pred)
            
            # Store attention weights
            if attention_weights is not None:
                attention_dict[bag_idx] = [aw.detach().cpu().numpy() for aw in attention_weights]
    
    return np.array(predictions), attention_dict


#Sklearn abMIL wrapper

class ABMILSklearnWrapper(BaseEstimator, ClassifierMixin):
    """Scikit-Learn wrapper for the custom ABMIL architecture."""
    
    def __init__(self, hidden_dim=256, attention_dim=128, dropout=0.2, 
                 learning_rate=1e-3, weight_decay=1e-5, batch_size=4, 
                 n_epochs=50, n_labels=5, device='cpu'):
        # Keep every parameter explicitly in __init__ for Scikit-Learn cloning mechanisms
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_labels = n_labels
        self.device = device
        
    def fit(self, X, y):
        """
        X : list of arrays (the bags)
        y : ndarray of shape (n_bags, n_labels)
        """
        # Call your native training script using the instance's tuned parameters
        self.model_, self.history_, self.scaler_ = train_abmil(
            X_bags_train=X,
            y_train=y,
            X_bags_val=None,
            y_val=None,
            n_labels=self.n_labels,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            hidden_dim=self.hidden_dim,
            attention_dim=self.attention_dim,
            dropout=self.dropout,
            label_specific_attention=True,
            device=self.device,
            verbose=False
        )
        # Standard scikit-learn convention requires returning self
        return self
        
    def predict_proba(self, X):
        """X : list of arrays (the test bags)"""
        # Delegate directly to your custom prediction module
        preds, _ = predict_abmil(self.model_, X, self.scaler_, device=self.device)
        return preds

    def predict(self, X):
        # Multi-label classification thresholding shortcut
        return (self.predict_proba(X) > 0.5).astype(int)
    
    # Define classes property to prevent scikit-learn multi-label metadata errors
    @property
    def classes_(self):
        return [np.array([0, 1]) for _ in range(self.n_labels)]


# ============================================================================
# USAGE 
# ============================================================================

def abmil_classifier_cv(X_bags,y,n_splits=5,random_state=42,n_labels=5): 
    label_names = ['Type A', 'Type B', 'Type C', 'Type D', 'Echo']
    y_np = np.array(y)
    kf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # 2. Setup Storage for metrics function
    y_true_all = []  # List of 5 arrays
    y_pred_all = []  # List of 5 arrays

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting Cross-Validation on Device: {device}")

    # 3. Cross-Validation Loop
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_bags, y_np)):
        print(f"\n{'='*30}")
        print(f" Processing Fold {fold + 1} / {n_splits} ")
        print(f"{'='*30}")

        # Split bags and labels for this fold
        X_bags_train = [X_bags[i] for i in train_idx]
        X_bags_test = [X_bags[i] for i in test_idx]
        y_train, y_test = y_np[train_idx], y_np[test_idx]

        # Train ABMIL (The function creates a fresh model instance internally)
        model, history, scaler = train_abmil(
            X_bags_train,
            y_train,
            X_bags_val=None, 
            y_val=None,
            n_labels=n_labels,
            n_epochs=100,
            batch_size=4,
            learning_rate=1e-3,
            weight_decay=1e-5,
            hidden_dim=256,
            attention_dim=128,
            dropout=0.2,
            label_specific_attention=True,
            device=device,
            verbose=False # Set to False to keep console clean during CV
        )

        # Predict on test set
        print(f"Evaluating Fold {fold + 1}...")
        y_pred_test, _ = predict_abmil(model, X_bags_test, scaler, device=device)

        # Store results
        y_true_all.append(y_test)
        y_pred_all.append(y_pred_test)

    return y_true_all, y_pred_all

def abmil_classifier_tuned(X_bags, y, n_split_out=5,n_split_in=5, num_trials=5,random_state=42):
    # 1. Initialize the split
    y = np.array(y) 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scorer = make_scorer(average_precision_score, average='macro', response_method='predict_proba')
    all_results = []

    for i in range(num_trials) :
        print(f"Starting Trial {i+1}/{num_trials} with random_state={random_state + i}...")
        model_params = {
            'ABMIL': {
                'model': ABMILSklearnWrapper(n_labels=y.shape[1],n_epochs=50, batch_size=4, device=device), #note : standard scaler already implemented
                'params': {
                    'hidden_dim': [128, 256],
                    'attention_dim': [64, 128],
                    'learning_rate': [1e-3, 5e-4]
                }
            }   
        }
        #Cross validation techniques for inner and outer loop
        inner_cv = MultilabelStratifiedKFold(n_splits=n_split_in, shuffle=True, random_state=random_state + i)
        outer_cv = MultilabelStratifiedKFold(n_splits=n_split_out, shuffle=True, random_state=random_state + i)

        #Nester CV with parameter optimisation for each model
        for model_name, mp in model_params.items():
            print(f"  Tuning and evaluating model: {model_name}")
            all_y_true = []
            all_y_pred_proba = []
            all_test_indices = []

            outer_scores = []

            for fold ,(train_idx, test_idx) in enumerate(outer_cv.split(X_bags, y)):
                print(f"    Evaluating fold {fold+1}/{n_split_out}")
                X_bags_train = [X_bags[i] for i in train_idx]
                X_bags_test = [X_bags[i] for i in test_idx]
                y_train, y_test = y[train_idx], y[test_idx]


                clf = GridSearchCV(estimator=mp['model'],param_grid=mp['params'],cv=inner_cv,
                                   scoring=scorer,refit=True,n_jobs=1)

                # fit on outer-train
                clf.fit(X_bags_train, y_train)

                # predict on outer-test
                y_pred_proba = clf.predict_proba(X_bags_test)

                #checking for y_pred_proba format and converting to [Samples, Labels] if needed
                if isinstance(y_pred_proba, list):
                    # For a list of arrays, extract the positive probability (column index 1) for each class
                    y_pred_proba = np.column_stack([prob[:, 1] for prob in y_pred_proba])
                elif isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 3:
                    # Alternative 3D representation sometimes returned by multi-output setups
                    y_pred_proba = y_pred_proba[:, :, 1].T

                # fold score
                fold_score = average_precision_score(y_test,y_pred_proba,average='macro')
                outer_scores.append(fold_score)

                # ---------------------------------
                # STORE OOF PREDICTIONS
                # ---------------------------------
                all_y_true.append(y_test)
                all_y_pred_proba.append(y_pred_proba)
                all_test_indices.append(test_idx)
            
            #Concatenate fold results to get out of fold predictions
            all_y_true = np.concatenate(all_y_true, axis=0)
            all_y_pred_proba = np.concatenate(all_y_pred_proba, axis=0)
            all_test_indices = np.concatenate(all_test_indices, axis=0)
            all_results.append({
                'trial': i,
                'model': model_name,

                'mean_AP': np.mean(outer_scores),
                'std_AP': np.std(outer_scores, ddof=1),

                'oof_y_true': all_y_true,
                'oof_y_pred_proba': all_y_pred_proba,
                'oof_indices': all_test_indices
            })
    
    # Return as numpy arrays for easier use in your compute_cv_stats
    return all_results
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    hamming_loss,
    accuracy_score,
    f1_score,
)
import warnings
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
        optimizer, mode='min', factor=0.5, patience=5, verbose=verbose
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


def evaluate_abmil(y_true, y_pred, y_pred_binary=None, threshold=0.5):
    """
    Evaluate ABMIL predictions.
    
    Parameters
    ----------
    y_true : (n_bags, n_labels) array
    y_pred : (n_bags, n_labels) array
        Probability predictions
    y_pred_binary : array or None
    threshold : float
    
    Returns
    -------
    metrics : dict
    """
    if y_pred_binary is None:
        y_pred_binary = (y_pred > threshold).astype(int)
    
    metrics = {
        "cmAP": average_precision_score(y_true, y_pred, average='macro'),
        "AUC-ROC (macro)": roc_auc_score(y_true, y_pred, average='macro'),
        "AUC-ROC (micro)": roc_auc_score(y_true, y_pred, average='micro'),
        "Hamming Loss": hamming_loss(y_true, y_pred_binary),
        "Subset Accuracy": accuracy_score(y_true, y_pred_binary),
        "F1 (macro)": f1_score(y_true, y_pred_binary, average='macro', zero_division=0),
    }
    
    # Per-label metrics
    metrics["AP per label"] = {}
    metrics["AUC per label"] = {}
    
    for i in range(y_true.shape[1]):
        metrics["AP per label"][f"Label {i}"] = average_precision_score(y_true[:, i], y_pred[:, i])
        metrics["AUC per label"][f"Label {i}"] = roc_auc_score(y_true[:, i], y_pred[:, i])
    
    return metrics


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    print("ABMIL module loaded successfully.")
    print("\nKey differences from mean-pooling:")
    print("  • Uses un-pooled window embeddings (one per window)")
    print("  • Learns importance weights via attention mechanism")
    print("  • Can identify which windows are diagnostic for each label")
    print("  • Often achieves better performance on MIL-suitable problems")

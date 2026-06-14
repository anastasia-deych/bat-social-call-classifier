"""
Multiple Instance Learning (MIL) for Multi-Label Bioacoustic Classification.
This module implements an Attention-Based Multiple Instance Learning (ABMIL) model for multi-label classification of bioacoustic recordings. 
Each recording is treated as a "bag" of instances (e.g., time windows or segments), 
and the model learns to predict multiple labels for the entire bag based on the features of its instances.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    make_scorer,
    roc_auc_score,
)
import warnings
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
warnings.filterwarnings('ignore')
import random


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
    1. Feature extraction : Maps instance features through FC layer
    2. Attention mechanism: Learns which instances are important
    3. Aggregation: Weighted sum using attention weights
    4. Classification: Multi-label head (sigmoid)
    5. Fall back to logistic regression for ultra-minority classes
    
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
            If True, use separate attention for each label
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
        
        self.classifier = nn.Linear(hidden_dim, n_labels)
            
    
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
    random_state = 42,
    ensemble = False,
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
    random_state : int
    ensemble : bool  decides whether to use Validation data for early stopping
    
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

    # Setup a deterministic generator for this specific execution context
    g = torch.Generator()
    g.manual_seed(random_state)
    
    # Create datasets
    train_dataset = MILDataset(X_bags_train, y_train, scaler=scaler, fit_scaler=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                collate_fn=collate_mil,
                                generator=g
                                )
    
    if X_bags_val is not None:
        val_dataset = MILDataset(X_bags_val, y_val, scaler=scaler, fit_scaler=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_mil,
                                generator=g
                                )
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
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if not ensemble :
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
    best_model_state = model.state_dict().copy() # Safe initialization fallback
    
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

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f} | "
                     f"Val Loss: {val_loss:.4f} | Val AP: {val_ap:.4f} | Val AUC: {val_auc:.4f}")
            
            if not ensemble : 
                # LR scheduling
                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
            
                if patience_counter >= 10:
                    print(f"Early stopping at epoch {epoch+1}")
                    model.load_state_dict(best_model_state)
                    break
        else:
            #  Pad validation arrays so shapes match during non-validation final fits
            history['val_loss'].append(np.nan)
            history['val_ap'].append(np.nan)
            history['val_auc'].append(np.nan)
    
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
            y_pred = torch.sigmoid(logits).detach().cpu().numpy()
            predictions.append(y_pred)
            
            # Store attention weights
            if attention_weights is not None:
                attention_dict[bag_idx] = [aw.detach().cpu().numpy() for aw in attention_weights]
    
    return np.array(predictions), attention_dict

class ABMILSklearnWrapper(BaseEstimator, ClassifierMixin):
    """
    SKLearn wrapper for ABMIL to enable use in GridSearchCV and RandomizedSearchCV.
    This wrapper handles training, prediction, and ultra-minority fallback logic for ABMIL.
    """
    def __init__(self, hidden_dim=256, attention_dim=128, dropout=0.2, 
                 learning_rate=1e-3, weight_decay=1e-5, batch_size=4, 
                 n_epochs=20, n_labels=5, device='cpu', random_state=42,
                 ensemble=False, ensemble_labels=None, lr_C=1.0):
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs  
        self.n_labels = n_labels
        self.device = device
        self.random_state = random_state
        self.ensemble = ensemble
        self.ensemble_labels = ensemble_labels  
        self.lr_C = lr_C

    def _get_ensemble_labels(self):
        """
        Safely retrieves the set of labels to ensemble on, defaulting to all labels if not specified.
        """
        return set(self.ensemble_labels) if self.ensemble_labels is not None else {0}

    def _meanpool(self, X):
        """
        Utility function to mean-pool bags for the logistic regression fallback.
        """
        return np.array([bag.mean(axis=0) for bag in X])
        
    def fit(self, X, y):
        """
        Fit the ABMIL model to the data.
            - X: list of arrays, each array is (n_windows, n_features) for one recording
            - y: (n_bags, n_labels) array of multi-label targets
        Fits ABMIL on the training data, and if ensembling is enabled, also fits logistic regression models for specified labels.
        """
        # Graceful handling if data is too small to split 5-fold internally

        # 1. Train the base ABMIL model (Fixed Epochs, NO Early Stopping)
        self.model_, self.history_, self.scaler_ = train_abmil(
            X_bags_train=X, y_train=y,
            n_labels=self.n_labels, n_epochs=self.n_epochs, batch_size=self.batch_size,
            learning_rate=self.learning_rate, weight_decay=self.weight_decay,
            hidden_dim=self.hidden_dim, attention_dim=self.attention_dim, dropout=self.dropout,
            label_specific_attention=True,
            device=self.device, verbose=False, random_state=self.random_state,
            ensemble = self.ensemble
        )

        # 2. If ensembling is active, build the base LR and stacker
        if self.ensemble:
            X_train_pool = self._meanpool(X) 
            self.lr_scaler_ = StandardScaler()
            X_train_pool_scaled = self.lr_scaler_.fit_transform(X_train_pool)

            self.base_lr_classifiers_ = {}
            for k in self._get_ensemble_labels():
                base_lr = LogisticRegression(C=self.lr_C, max_iter=1000, random_state=self.random_state)
                base_lr.fit(X_train_pool_scaled, y[:, k])
                self.base_lr_classifiers_[k] = base_lr

        return self
        
    def predict_proba(self, X):
        """Predict probabilities for the given data."""
        abmil_preds, _ = predict_abmil(self.model_, X, self.scaler_, device=self.device)

        if not self.ensemble:
            return abmil_preds

        X_pool = self._meanpool(X)
        X_pool_scaled = self.lr_scaler_.transform(X_pool)

        preds = abmil_preds.copy()
        for k in self._get_ensemble_labels():
            base_lr_pred = self.base_lr_classifiers_[k].predict_proba(X_pool_scaled)[:, 1]
            #meta_features_test = np.column_stack([abmil_preds[:, k], base_lr_pred])
            preds[:, k] = base_lr_pred

        return preds

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)
    
    @property
    def classes_(self):
        return [np.array([0, 1]) for _ in range(self.n_labels)]

# ============================================================================
# USAGE 
# ============================================================================
def abmil_classifier_tuned(X_bags, y, n_split_out=5,n_split_in=5, num_trials=5,random_state=42,
                           n_iter_search = 4):
    """
    Trains and evaluates ABMIL with nested cross-validation and hyperparameter tuning.
    X_bags: list of arrays, each array is (n_windows, n_features) for one recording
    y: (n_bags, n_labels) array of multi-label targets
    n_iter_search: int, number of parameter settings that are sampled in RandomizedSearchCV for each fold
    n_split_out: int, number of splits for outer cross-validation
    n_split_in: int, number of splits for inner cross-validation
    num_trials: int, number of repeated nested CV runs with different random seeds for robustness
    random_state: int, base random seed for reproducibility
    Returns a list of results dictionaries containing OOF predictions, best models, and performance metrics for each trial.
    """
     
    # 1. Initialize the split
    y = np.array(y) 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scorer = make_scorer(average_precision_score, average='macro', response_method='predict_proba')
    all_results = []

    for i in range(num_trials) :
        trial_seed = random_state + i
        print(f"Starting Trial {i+1}/{num_trials} with random_state={trial_seed}...")
        
        #set random seeds
        np.random.seed(trial_seed)
        random.seed(trial_seed)
        torch.manual_seed(trial_seed)
        torch.cuda.manual_seed_all(trial_seed)

        model_params = {
            'ABMIL': {
                'model': ABMILSklearnWrapper(n_labels=y.shape[1],n_epochs=20, batch_size=4,
                                             device=device,ensemble = True,ensemble_labels=[0],random_state=trial_seed),
                'params': {
                    'hidden_dim': [64, 128, 256],
                    'attention_dim': [32, 64, 128],
                    'dropout': [0.1, 0.3, 0.5],
                    'learning_rate': loguniform(1e-4, 5e-3),
                    'weight_decay': loguniform(1e-6, 1e-2),
                    'lr_C': [0.1, 1.0, 10.0],
                }
            }   
        }
         #Cross validation techniques for inner and outer loop
        inner_cv = MultilabelStratifiedKFold(n_splits=n_split_in, shuffle=True, random_state=trial_seed)
        outer_cv = MultilabelStratifiedKFold(n_splits=n_split_out, shuffle=True, random_state=trial_seed)

        #Nester CV with parameter optimisation for each model
        for model_name, mp in model_params.items():
            print(f"  Tuning and evaluating model: {model_name}")
            all_y_true = []
            all_y_pred_proba = []
            all_test_indices = []
            best_models = []

            outer_scores = []
            fold_train_histories = []
            fold_val_histories = []

            for fold ,(train_idx, test_idx) in enumerate(outer_cv.split(X_bags, y)):
                print(f"    Evaluating fold {fold+1}/{n_split_out}")
                X_bags_train = [X_bags[i] for i in train_idx]
                X_bags_test = [X_bags[i] for i in test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                clf = RandomizedSearchCV(
                    estimator=mp['model'],
                    param_distributions=mp['params'],
                    n_iter=n_iter_search, # Controls your exact compute budget per fold
                    cv=inner_cv,
                    scoring=scorer,
                    refit=True,
                    n_jobs=1,
                    random_state=trial_seed*100+fold,
                    verbose=3
                )       

                # fit on outer-train
                clf.fit(X_bags_train, y_train)
                #  extract history metrics from the refitted optimal estimator instance
                best_model_instance = clf.best_estimator_
                fold_train_histories.append(best_model_instance.history_['train_loss'])
                fold_val_histories.append(best_model_instance.history_['val_loss'])

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
                best_models.append(clf.best_estimator_)
            
            #Concatenate fold results to get out of fold predictions
            y_true_cv = all_y_true
            y_pred_proba_cv = all_y_pred_proba
            all_y_true = np.concatenate(all_y_true, axis=0)
            all_y_pred_proba = np.concatenate(all_y_pred_proba, axis=0)
            all_test_indices = np.concatenate(all_test_indices, axis=0)
            all_results.append({
                'trial': i,
                'model': model_name,
                'best_models' : best_models,

                'mean_AP': np.mean(outer_scores),
                'std_AP': np.std(outer_scores, ddof=1),

                'y_true_cv' : y_true_cv,
                'y_pred_proba_cv' : y_pred_proba_cv,
                'oof_y_true': all_y_true,
                'oof_y_pred_proba': all_y_pred_proba,
                'oof_indices': all_test_indices,
                'train_histories': fold_train_histories,
                'val_histories': fold_val_histories
            })
    
    # Return as numpy arrays for easier use in your compute_cv_stats
    return all_results


def analyze_best_hyperparameters(abmil_results):
    """
    Analyzes the best hyperparameters from a list of ABMIL results.
    abmil_results: list of dicts, each containing 'best_models' which is a list of fitted ABMILSklearnWrapper instances from each fold and trial.
    """

    all_params = []
    
    # 1. Loop through all runs and extract parameters from the best models
    for run_idx, run in enumerate(abmil_results):
        if 'best_models' in run:
            for model in run['best_models']:
                # Strategy A: Use scikit-learn standard get_params if available
                if hasattr(model, 'get_params'):
                    params = model.get_params(deep=False)
                # Strategy B: Check if it directly exposes a best_params_ dict
                elif hasattr(model, 'best_params_'):
                    params = model.best_params_
                # Strategy C: Fall back to pulling primitive attributes from the object instance
                else:
                    params = {k: v for k, v in model.__dict__.items() 
                              if isinstance(v, (int, float, str, bool, list, tuple, type(None))) 
                              and not k.startswith('_')}
                
                all_params.append(params)
                
    if not all_params:
        print("No hyperparameters found. Please verify the structure of your abmil_results.")
        return None

    # 2. Convert to DataFrame
    df_params = pd.DataFrame(all_params)
    
    # 3. Clean up non-hyperparameters (like system paths, device configurations, or objects)
    cols_to_drop = ['device', 'scaler', 'model_'] # add any other non-tuning columns here if needed
    cols_to_drop = [c for c in cols_to_drop if c in df_params.columns]
    df_params = df_params.drop(columns=cols_to_drop)
    
    # 4. Make lists hashable (e.g., converting [128, 64] to (128, 64)) so value_counts works
    for col in df_params.columns:
        df_params[col] = df_params[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    # 5. Drop completely identical columns (parameters that never changed during search)
    # to keep the output clean and readable
    varying_cols = [col for col in df_params.columns if df_params[col].nunique() > 1]
    
    if not varying_cols:
        print("All search parameters were identical across every run!\n")
        print(df_params.iloc[0].to_frame(name='Value'))
        return df_params

    df_varying = df_params[varying_cols]

    # 6. Calculate frequencies
    print("=" * 60)
    print(" MOST FREQUENT WINNING COMBINATIONS ")
    print("=" * 60)
    top_combinations = df_varying.value_counts().reset_index(name='Occurrence Count')
    print(top_combinations.to_string(index=False))
    print("\n" + "=" * 60)
    
    print(" MOST FREQUENT INDIVIDUAL PARAMETERS (MODE) ")
    print("=" * 60)
    for col in df_varying.columns:
        mode_val = df_varying[col].mode()[0]
        mode_count = (df_varying[col] == mode_val).sum()
        print(f" -> {col}: Most frequent choice was {mode_val} (found {mode_count}/{len(df_params)} times)")
        
    return top_combinations

def abmil_classifier_deployement(X_bags, y, n_split=5, random_state=42):
    """
    Trains ABMIL in non nested cv for tuning a restricted hyperparameter grid deduced from the previous nested CV results.
    Done for finding the best hyperparameters to train a final model on 100% of the data for deployment and OOD evaluation.
    X_bags: list of arrays, each array is (n_windows, n_features) for one recording
    y: (n_bags, n_labels) array of multi-label targets
    """

    # 1. Initialize the dataset and environment
    y = np.array(y) 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scorer = make_scorer(average_precision_score, average='macro', response_method='predict_proba')
    all_results = []
    
    ensemble = True
    ensemble_labels = [0]

    trial_seed = random_state
    
    # Set random seeds
    np.random.seed(trial_seed)
    random.seed(trial_seed)
    torch.manual_seed(trial_seed)
    torch.cuda.manual_seed_all(trial_seed)

    model_params = {
        'ABMIL': {
            'model': ABMILSklearnWrapper(
                n_labels=y.shape[1], n_epochs=20, batch_size=4,
                device=device,
                ensemble=ensemble, ensemble_labels=ensemble_labels, 
                random_state=trial_seed
            ),
            'params': {
                'hidden_dim': [256],
                'attention_dim': [128],
                'dropout': [0.1],
                # FIXED: Converted loguniform distributions to concrete lists for Grid Search
                'learning_rate': [1e-4, 3e-4, 1e-3],
                'weight_decay': [1e-6, 1e-4, 0.007],
                **({'lr_C': [0.1]} if ensemble else {})
            }
        }   
    }

    # Cross-validation technique for the inner optimization loops
    cv = MultilabelStratifiedKFold(n_splits=n_split, shuffle=True, random_state=trial_seed)
    
    for model_name, mp in model_params.items():
        print(f" Tuning and evaluating model: {model_name}")
        best_models = []
        fold_train_histories = []
        fold_val_histories = []

        clf = GridSearchCV(
            estimator=mp['model'],
            param_grid=mp['params'], 
            cv=cv,
            scoring=scorer,
            refit=True,
            n_jobs=1,
            verbose=3
        )  
        
        # Fit on the entire dataset. 
        # It uses CV to find the best configuration, then refits ONE final model on 100% of the data.
        clf.fit(X_bags, y)

        # Extract history metrics from the final refitted optimal estimator instance
        best_model_instance = clf.best_estimator_
        
        # Note: These will contain a single history path belonging to the final refitted model
        if hasattr(best_model_instance, 'history_'):
            fold_train_histories.append(best_model_instance.history_.get('train_loss', []))
            fold_val_histories.append(best_model_instance.history_.get('val_loss', []))
        
        best_models.append(best_model_instance)
    
        all_results.append({
            'model': model_name,
            'best_models': best_models,
            'train_histories': fold_train_histories,
            'val_histories': fold_val_histories,
            'best_params': clf.best_params_,
            'best_score': clf.best_score_
        })
    
    return all_results

def train_abmil_all_data(X_bags, y, random_state=42):
    """
    Trains ABMIL on all the data using best hyperparameters found from previous tuning step
    X_bags: list of arrays, each array is (n_windows, n_features) for one recording
    y: (n_bags, n_labels) array of multi-label targets
    """
    # 1. Initialize the dataset and environment
    y = np.array(y) 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ensemble = True
    ensemble_labels = [0]

    trial_seed = random_state
    
    # Set random seeds
    np.random.seed(trial_seed)
    random.seed(trial_seed)
    torch.manual_seed(trial_seed)
    torch.cuda.manual_seed_all(trial_seed)

    clf =ABMILSklearnWrapper(
                n_labels=y.shape[1], n_epochs=20, batch_size=4,
                device=device, 
                ensemble=ensemble, ensemble_labels=ensemble_labels, 
                random_state=trial_seed, hidden_dim=256,attention_dim=128,dropout=0.1,
                learning_rate= 1e-4,weight_decay=1e-6,lr_C = 0.1
            )

        
    # Fit directly on 100% of the dataset using your best hyperparameters
    print("Training production model on all available data...")
    clf.fit(X_bags, y)
    
    return clf

def evaluate_abmil_ood(fitted_wrapper,X_bags_ood) :
    """
    Evaluates the fitted ABMIL model on OOD data.
     - fitted_wrapper: The ABMILSklearnWrapper instance that has been trained on all data
    - X_bags_ood: List of arrays for OOD recordings, list of (n_windows, n_features) arrays
    """
    # 2. Extract the underlying PyTorch items required by your predict_abmil function
    pt_model = fitted_wrapper.model_   # The trained neural net
    scaler = fitted_wrapper.scaler_     # The standard scaler calibrated to your data training distribution
    device = fitted_wrapper.device     # GPU or CPU flag

    # 3. Put PyTorch model into evaluation mode
    pt_model.eval()

    # 4. Score your OOD data
    y_pred_proba_ood, attention_weights_ood = predict_abmil(
        pt_model, 
        X_bags_ood, 
        scaler, 
        device=device
    )
    return y_pred_proba_ood, attention_weights_ood
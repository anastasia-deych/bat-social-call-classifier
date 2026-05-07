
from turtle import pd

import numpy as np

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.base import BaseEstimator, ClassifierMixin

from preprocessing.dataset import PipistrelleDataset
from models.feature_generation import build_feature_bank, extract_encoder

def linear_probe_online(csv_data : str, n_split=5, random_state=42,balance : bool = False,encoder_name : str = 'perch2'):
    #0. Initialise dataset 
    df = pd.read_csv(csv_data)
    label_cols = ['type_a', 'type_b', 'type_c', 'type_d', 'echo']
    # 1. Initialize the split
    kf = MultilabelStratifiedKFold(n_splits=n_split, shuffle=True, random_state=random_state)
    clf_names = ['SVM', 'Random Forest', 'MLP']

    # 2. Setup storage
    y_true_all = [] # Will hold one y_test per fold (length = n_split)
    y_proba_all = {name: [] for name in clf_names}

    #3. Extract Encoder 
    encoder = extract_encoder(encoder_name, device='cpu')

    for fold, (train_idx, test_idx) in enumerate(kf.split(df, df[label_cols].values)):
        print(f"Processing Fold {fold+1}...")

        # 1. Instantiate Datasets for this fold
        train_ds = PipistrelleDataset(df.iloc[train_idx], is_training=True, resample=True)
        test_ds = PipistrelleDataset(df.iloc[test_idx], is_training=False, resample=False)

        # 2. Extract Static Features for SVM/RF (One pass through the dataset)
        # This gives SVM/RF one augmented version of the training data
        X_train_static, y_train_static = build_feature_bank(train_ds, encoder, encoder_name, device='cpu')
        X_test_static, y_test_static = build_feature_bank(test_ds, encoder, encoder_name, device='cpu')

        # 3. Scaler (Fit on training snapshot)
        scaler = StandardScaler()
        X_train_static = scaler.fit_transform(X_train_static)
        X_test_static = scaler.transform(X_test_static)
        
        y_true_all.append(y_test_static)

        models = {
            'SVM': OneVsRestClassifier(SVC(
                probability=True, 
                random_state=random_state,
                class_weight='balanced' if balance else None)),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=random_state,
                class_weight='balanced' if balance else None), # RF is natively multi-label
             #Random Forest': OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=random_state)),
            'MLP': BalancedMLP(
                input_dim=X_train_static.shape[1],
                hidden_dim=128,
                lr=0.001,
                epochs=50,
                dropout=0.2,
                balanced=balance,
                batch_norm=True
            )
        }

        for name, clf in models.items():
            if name == 'MLP':
                train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
                clf.fit_with_loader(train_loader, epochs=50) # Custom fit method
                y_proba_all['MLP'].append(clf.predict_proba(X_test_static))
            else :
                clf.fit(X_train_static, y_train_static)
                y_proba = clf.predict_proba(X_test_static)
            
                 # predict_proba for multi-label often returns a list of arrays
                # We want to ensure it's a consistent [Samples, Labels] array
                if isinstance(y_proba, list):
                    # Convert list of [Samples, 2] to [Samples, Labels] using the positive class proba
                    y_proba = np.array([p[:, 1] for p in y_proba]).T
            
                y_proba_all[name].append(y_proba)
    
    # Return as numpy arrays for easier use in your compute_cv_stats
    return y_true_all, y_proba_all




def linear_probe(X, y, n_split=5, random_state=42,balance : bool = False):
    # 1. Initialize the split
    kf = MultilabelStratifiedKFold(n_splits=n_split, shuffle=True, random_state=random_state)
    clf_names = ['SVM', 'Random Forest', 'MLP']

    # 2. Setup storage
    y_true_all = [] # Will hold one y_test per fold (length = n_split)
    y_proba_all = {name: [] for name in clf_names}

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        print(f"Processing Fold {fold+1}...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 3. CRITICAL: Scale features for SVM and MLP
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Store y_test ONCE per fold
        y_true_all.append(y_test)

        models = {
            'SVM': OneVsRestClassifier(SVC(
                probability=True, 
                random_state=random_state,
                class_weight='balanced' if balance else None)),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=random_state,
                class_weight='balanced' if balance else None), # RF is natively multi-label
             #Random Forest': OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=random_state)),
            'MLP': BalancedMLP(
                input_dim=X.shape[1],
                hidden_dim=128,
                lr=0.001,
                epochs=50,
                dropout=0.2,
                balanced=balance,
                batch_norm=False
            )
        }

        for name, clf in models.items():
            clf.fit(X_train, y_train)
            y_proba = clf.predict_proba(X_test)
            
            # predict_proba for multi-label often returns a list of arrays
            # We want to ensure it's a consistent [Samples, Labels] array
            if isinstance(y_proba, list):
                # Convert list of [Samples, 2] to [Samples, Labels] using the positive class proba
                y_proba = np.array([p[:, 1] for p in y_proba]).T
            
            y_proba_all[name].append(y_proba)
    
    # Return as numpy arrays for easier use in your compute_cv_stats
    return y_true_all, y_proba_all



class BalancedMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=1536, hidden_dim=128, lr=0.001, epochs=50, dropout=0.2,balanced = False, batch_norm : bool = False):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.model = None
        self.classes_ = None
        self.balanced : bool = balanced
        self.batch_norm : bool = batch_norm
    def _build_model(self, output_dim):
        return nn.Sequential(
            nn.BatchNorm1d(self.input_dim) if self.batch_norm else nn.Identity(),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def fit(self, X, y):
        # Convert to Tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # 1. Handle Class Imbalance Automatically
        # pos_weight = (count_negative / count_positive)
        num_pos = y_tensor.sum(dim=0)
        num_neg = y_tensor.size(0) - num_pos
        # Add small epsilon to avoid division by zero
        pos_weight = num_neg / (num_pos + 1e-6) 
        
        # 2. Setup Training
        self.model = self._build_model(y.shape[1])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight if self.balanced else None)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 3. Training Loop
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
        self.classes_ = np.arange(y.shape[1])
        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            logits = self.model(X_tensor)
            # BCEWithLogitsLoss outputs logits; sigmoid turns them into 0-1 probabilities
            probs = torch.sigmoid(logits).numpy()
        return probs

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs > threshold).astype(int)  
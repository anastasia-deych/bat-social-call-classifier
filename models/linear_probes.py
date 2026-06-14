

import pandas as pd

import numpy as np
import random

from sklearn.metrics import average_precision_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.dummy import DummyClassifier


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from models.classifier_models import BalancedMLP

import torch
import torch.nn as nn
import torch.optim as optim

def linear_probe_tuned(X, y, n_split_out=5,n_split_in=5, num_trials=5,random_state=42,balance : bool = False):
    """
    Performs nested cross-validation with hyperparameter tuning for multiple classifiers.
    - X: Feature matrix (numpy array).
    - y: Multi-label binary target matrix (numpy array).
    - n_split_out: Number of splits for the outer loop (model evaluation).
    - n_split_in: Number of splits for the inner loop (hyperparameter tuning).
    - num_trials: Number of repeated trials with different random seeds for robustness.
    - random_state: Base random seed for reproducibility.
    - balance: Whether to use class_weight='balanced' for applicable models.

    Returns a list of results containing true labels, predicted probabilities, and performance metrics for each model
    """
    # 1. Initialize the split
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

        #instantiate model configurations
        model_params = {
            'SVM': {
                'model': OneVsRestClassifier(SVC(
                            probability=True, 
                            random_state=trial_seed, # Updated dynamically
                            class_weight='balanced' if balance else None)),
                'params': {
                    'model__estimator__C': [1, 10, 20],
                    'model__estimator__kernel': ['rbf', 'linear'],
                    'model__estimator__gamma': ['scale', 'auto', 0.01, 0.1]
                }
            }, 
            'Logistic Regression': {
                'model': OneVsRestClassifier(LogisticRegression(
                            max_iter=1000,             
                            random_state=trial_seed,     
                            class_weight='balanced' if balance else None
                         )),
                'params': {
                    'model__estimator__C': [0.1, 1.0, 10.0],       # Standard inverse regularization strengths
                    'model__estimator__solver': ['lbfgs']   #  multi-label optimization solvers
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(
                            n_estimators=100, 
                            random_state=trial_seed, # Updated dynamically
                            class_weight='balanced' if balance else None), 
                'params': {
                    'model__n_estimators': [100],
                    'model__max_depth': [None, 10, 20]
                }
            },
            'MLP' : {
                'model' : BalancedMLP(
                    input_dim=X.shape[1],
                    hidden_dim=128,
                    lr=0.001,
                    epochs=50,
                    dropout=0.2,
                    balanced=balance,
                    batch_norm=False,
                    random_state=trial_seed
                ),
                'params' : {
                    'model__lr':[0.001],
                    'model__hidden_dim':[128],
                    'model__epochs':[50],
                    'model__dropout':[0.2, 0.5]
                }
            },
            'Prevalence guesser': {
                # Ensure your stochastic baseline uses NumPy or accepts a random_state seed!
                #'model': MultilabelPrevalenceBaseline(type='stochastic', random_state=trial_seed),
                'model' : DummyClassifier(strategy='stratified', random_state=trial_seed),
                'params': {}
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

            outer_scores = []

            for fold ,(train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
                print(f"    Evaluating fold {fold+1}/{n_split_out}")
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', mp['model'])
                ])

                clf = GridSearchCV(estimator=pipeline,param_grid=mp['params'],cv=inner_cv,
                                   scoring=scorer,refit=True,n_jobs=1 if model_name == 'MLP' else -1)

                # fit on outer-train
                clf.fit(X_train, y_train)

                # predict on outer-test
                y_pred_proba = clf.predict_proba(X_test)

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
            y_true_cv = all_y_true
            y_pred_proba_cv = all_y_pred_proba
            all_y_true = np.concatenate(all_y_true, axis=0)
            all_y_pred_proba = np.concatenate(all_y_pred_proba, axis=0)
            all_test_indices = np.concatenate(all_test_indices, axis=0)
            all_results.append({
                'trial': i,
                'model': model_name,

                'mean_AP': np.mean(outer_scores),
                'std_AP': np.std(outer_scores, ddof=1),

                'y_true_cv' : y_true_cv,
                'y_pred_proba_cv' : y_pred_proba_cv,
                'oof_y_true': all_y_true,
                'oof_y_pred_proba': all_y_pred_proba,
                'oof_indices': all_test_indices
            })
    
    # Return as numpy arrays for easier use in your compute_cv_stats
    return all_results


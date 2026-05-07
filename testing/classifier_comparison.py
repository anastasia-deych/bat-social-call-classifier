"""
Multi-Classifier Comparison: SVM, MLP, Random Forest for One-vs-Rest Multi-Label Classification.

Compares three classifiers:
1. SVM (Support Vector Machine) with RBF kernel
2. MLP (Multi-Layer Perceptron)
3. Random Forest

Metrics:
- Macro-AUC (average AUC across all labels)
- Per-label PR-AUC (Precision-Recall Area Under Curve)
- cmAP (class-wise mean Average Precision)
- Brier Score (mean squared error) per class
- Log-Loss per class
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc,
    log_loss,
    brier_score_loss,
    hamming_loss,
    accuracy_score,
    f1_score,
)
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_svm_ovr(X_train, y_train, label_names, C=1.0, kernel='rbf', verbose=True):
    """
    Train SVM classifiers using One-vs-Rest approach.
    
    Parameters
    ----------
    X_train : (n_samples, n_features) array
    y_train : (n_samples, n_labels) binary array
    label_names : list of str
    C : float
        Regularization parameter
    kernel : str
        'rbf', 'linear', 'poly'
    verbose : bool
    
    Returns
    -------
    svm_models : dict
        {label_index: trained SVC model}
    """
    n_labels = y_train.shape[1]
    svm_models = {}
    
    for i, label_name in enumerate(label_names):
        if verbose:
            print(f"  Training SVM for {label_name}...")
        
        y_binary = y_train[:, i]
        
        svm = SVC(
            C=C,
            kernel=kernel,
            probability=True,  # Enable probability estimation
            random_state=42,
        )
        svm.fit(X_train, y_binary)
        svm_models[i] = svm
    
    return svm_models


def train_mlp_ovr(X_train, y_train, label_names, hidden_layer_sizes=(256, 128), 
                  learning_rate=1e-3, max_iter=1000, verbose=True):
    """
    Train MLP classifiers using One-vs-Rest approach.
    
    Parameters
    ----------
    X_train : (n_samples, n_features) array
    y_train : (n_samples, n_labels) binary array
    label_names : list of str
    hidden_layer_sizes : tuple
    learning_rate : float
    max_iter : int
    verbose : bool
    
    Returns
    -------
    mlp_models : dict
        {label_index: trained MLPClassifier model}
    """
    n_labels = y_train.shape[1]
    mlp_models = {}
    
    for i, label_name in enumerate(label_names):
        if verbose:
            print(f"  Training MLP for {label_name}...")
        
        y_binary = y_train[:, i]
        
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            verbose=False,
        )
        mlp.fit(X_train, y_binary)
        mlp_models[i] = mlp
    
    return mlp_models


def train_rf_ovr(X_train, y_train, label_names, n_estimators=100, 
                 max_depth=None, verbose=True):
    """
    Train Random Forest classifiers using One-vs-Rest approach.
    
    Parameters
    ----------
    X_train : (n_samples, n_features) array
    y_train : (n_samples, n_labels) binary array
    label_names : list of str
    n_estimators : int
    max_depth : int or None
    verbose : bool
    
    Returns
    -------
    rf_models : dict
        {label_index: trained RandomForestClassifier model}
    """
    n_labels = y_train.shape[1]
    rf_models = {}
    
    for i, label_name in enumerate(label_names):
        if verbose:
            print(f"  Training Random Forest for {label_name}...")
        
        y_binary = y_train[:, i]
        
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_binary)
        rf_models[i] = rf
    
    return rf_models


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_ovr(models, X_test, return_proba=True):
    """
    Get predictions from OvR models.
    
    Parameters
    ----------
    models : dict
        {label_index: trained model}
    X_test : (n_samples, n_features) array
    return_proba : bool
        If True, return probabilities; else binary predictions
    
    Returns
    -------
    predictions : (n_samples, n_labels) array
        Probabilities or binary predictions
    """
    n_labels = len(models)
    n_samples = X_test.shape[0]
    
    predictions = np.zeros((n_samples, n_labels))
    
    for i in range(n_labels):
        if return_proba:
            # Get probability of positive class
            proba = models[i].predict_proba(X_test)[:, 1]
            predictions[:, i] = proba
        else:
            predictions[:, i] = models[i].predict(X_test)
    
    return predictions


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_comprehensive_metrics(y_true, y_pred_proba, y_pred_binary=None, 
                                  label_names=None, threshold=0.5):
    """
    Compute comprehensive multi-label metrics.
    
    Parameters
    ----------
    y_true : (n_samples, n_labels) array
    y_pred_proba : (n_samples, n_labels) array
        Probability predictions
    y_pred_binary : array or None
    label_names : list of str or None
    threshold : float
    
    Returns
    -------
    metrics : dict
        Comprehensive metrics
    """
    if y_pred_binary is None:
        y_pred_binary = (y_pred_proba > threshold).astype(int)
    
    n_labels = y_true.shape[1]
    if label_names is None:
        label_names = [f"Label {i}" for i in range(n_labels)]
    
    metrics = {}
    
    # ========================================================================
    # INSTANCE-LEVEL METRICS
    # ========================================================================
    metrics['Hamming Loss'] = hamming_loss(y_true, y_pred_binary)
    metrics['Subset Accuracy'] = accuracy_score(y_true, y_pred_binary)
    
    # ========================================================================
    # LABEL-LEVEL METRICS (MACRO)
    # ========================================================================
    metrics['F1 (macro)'] = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
    metrics['F1 (micro)'] = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
    
    # ========================================================================
    # RANKING METRICS
    # ========================================================================
    metrics['cmAP'] = average_precision_score(y_true, y_pred_proba, average='macro')
    metrics['Macro-AUC'] = roc_auc_score(y_true, y_pred_proba, average='macro')
    metrics['Micro-AUC'] = roc_auc_score(y_true, y_pred_proba, average='micro')
    
    # ========================================================================
    # PER-LABEL METRICS
    # ========================================================================
    metrics['AP per label'] = {}
    metrics['AUC per label'] = {}
    metrics['PR-AUC per label'] = {}
    metrics['Brier per label'] = {}
    metrics['Log-Loss per label'] = {}
    
    for i, label_name in enumerate(label_names):
        # Average Precision
        ap = average_precision_score(y_true[:, i], y_pred_proba[:, i])
        metrics['AP per label'][label_name] = ap
        
        # AUC-ROC
        auc_roc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
        metrics['AUC per label'][label_name] = auc_roc
        
        # PR-AUC (Precision-Recall AUC)
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred_proba[:, i])
        pr_auc = auc(recall, precision)
        metrics['PR-AUC per label'][label_name] = pr_auc
        
        # Brier Score (lower is better)
        brier = brier_score_loss(y_true[:, i], y_pred_proba[:, i])
        metrics['Brier per label'][label_name] = brier
        
        # Log-Loss (lower is better)
        ll = log_loss(y_true[:, i], y_pred_proba[:, i])
        metrics['Log-Loss per label'][label_name] = ll
    
    # Macro averages for per-label metrics
    metrics['Brier (macro)'] = np.mean(list(metrics['Brier per label'].values()))
    metrics['Log-Loss (macro)'] = np.mean(list(metrics['Log-Loss per label'].values()))
    metrics['PR-AUC (macro)'] = np.mean(list(metrics['PR-AUC per label'].values()))
    
    return metrics


# ============================================================================
# COMPARISON FUNCTION
# ============================================================================

def compare_classifiers_cv(X, y, label_names, n_splits=5, random_state=42):
    """
    Train and compare classifiers using 5-fold cross-validation.
    
    Parameters
    ----------
    X : (n_samples, n_features) array
    y : (n_samples, n_labels) binary array
    label_names : list of str
    n_splits : int
        Number of CV folds
    random_state : int
    
    Returns
    -------
    results : dict
        All results with CV statistics
    """
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    results = {
        'n_splits': n_splits,
        'classifiers_per_fold': {
            'SVM': [],
            'MLP': [],
            'Random Forest': [],
        },
        'metrics_per_fold': {
            'SVM': [],
            'MLP': [],
            'Random Forest': [],
        },
        'cv_stats': {
            'SVM': {},
            'MLP': {},
            'Random Forest': {},
        },
    }
    
    fold_idx = 0
    
    for train_idx, test_idx in kf.split(X):
        fold_idx += 1
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}/{n_splits}")
        print(f"{'='*60}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # ====================================================================
        # SVM
        # ====================================================================
        print("Training SVM...")
        svm_models = train_svm_ovr(X_train_scaled, y_train, label_names, verbose=False)
        y_pred_svm_proba = predict_ovr(svm_models, X_test_scaled, return_proba=True)
        y_pred_svm_binary = predict_ovr(svm_models, X_test_scaled, return_proba=False)
        
        metrics_svm = compute_comprehensive_metrics(
            y_test, y_pred_svm_proba, y_pred_binary=y_pred_svm_binary, label_names=label_names
        )
        
        results['classifiers_per_fold']['SVM'].append(svm_models)
        results['metrics_per_fold']['SVM'].append(metrics_svm)
        
        # ====================================================================
        # MLP
        # ====================================================================
        print("Training MLP...")
        mlp_models = train_mlp_ovr(X_train_scaled, y_train, label_names, verbose=False)
        y_pred_mlp_proba = predict_ovr(mlp_models, X_test_scaled, return_proba=True)
        y_pred_mlp_binary = predict_ovr(mlp_models, X_test_scaled, return_proba=False)
        
        metrics_mlp = compute_comprehensive_metrics(
            y_test, y_pred_mlp_proba, y_pred_binary=y_pred_mlp_binary, label_names=label_names
        )
        
        results['classifiers_per_fold']['MLP'].append(mlp_models)
        results['metrics_per_fold']['MLP'].append(metrics_mlp)
        
        # ====================================================================
        # RANDOM FOREST
        # ====================================================================
        print("Training Random Forest...")
        rf_models = train_rf_ovr(X_train_scaled, y_train, label_names, verbose=False)
        y_pred_rf_proba = predict_ovr(rf_models, X_test_scaled, return_proba=True)
        y_pred_rf_binary = predict_ovr(rf_models, X_test_scaled, return_proba=False)
        
        metrics_rf = compute_comprehensive_metrics(
            y_test, y_pred_rf_proba, y_pred_binary=y_pred_rf_binary, label_names=label_names
        )
        
        results['classifiers_per_fold']['Random Forest'].append(rf_models)
        results['metrics_per_fold']['Random Forest'].append(metrics_rf)
    
    # ========================================================================
    # COMPUTE CV STATISTICS
    # ========================================================================
    print(f"\n{'='*60}")
    print("Computing CV statistics...")
    print(f"{'='*60}")
    
    metrics_to_aggregate = [
        'Macro-AUC', 'cmAP', 'PR-AUC (macro)', 'Brier (macro)', 'Log-Loss (macro)', 'F1 (macro)',
        'Subset Accuracy', 'Hamming Loss'
    ]
    
    for clf_name in ['SVM', 'MLP', 'Random Forest']:
        cv_stats = results['cv_stats'][clf_name]
        
        for metric_name in metrics_to_aggregate:
            values = [results['metrics_per_fold'][clf_name][fold][metric_name] 
                     for fold in range(n_splits)]
            
            cv_stats[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values,
            }
        
        # Per-label metrics
        cv_stats['AP per label'] = {}
        cv_stats['PR-AUC per label'] = {}
        cv_stats['Brier per label'] = {}
        cv_stats['Log-Loss per label'] = {}
        
        for label_name in label_names:
            ap_values = [results['metrics_per_fold'][clf_name][fold]['AP per label'][label_name]
                        for fold in range(n_splits)]
            cv_stats['AP per label'][label_name] = {
                'mean': np.mean(ap_values),
                'std': np.std(ap_values),
                'min': np.min(ap_values),
                'max': np.max(ap_values),
            }
            
            pr_values = [results['metrics_per_fold'][clf_name][fold]['PR-AUC per label'][label_name]
                        for fold in range(n_splits)]
            cv_stats['PR-AUC per label'][label_name] = {
                'mean': np.mean(pr_values),
                'std': np.std(pr_values),
                'min': np.min(pr_values),
                'max': np.max(pr_values),
            }
            
            brier_values = [results['metrics_per_fold'][clf_name][fold]['Brier per label'][label_name]
                           for fold in range(n_splits)]
            cv_stats['Brier per label'][label_name] = {
                'mean': np.mean(brier_values),
                'std': np.std(brier_values),
                'min': np.min(brier_values),
                'max': np.max(brier_values),
            }
            
            ll_values = [results['metrics_per_fold'][clf_name][fold]['Log-Loss per label'][label_name]
                        for fold in range(n_splits)]
            cv_stats['Log-Loss per label'][label_name] = {
                'mean': np.mean(ll_values),
                'std': np.std(ll_values),
                'min': np.min(ll_values),
                'max': np.max(ll_values),
            }
    
    return results


# ============================================================================
# RESULTS SUMMARY
# ============================================================================

def print_results_summary(results, label_names):
    """Print comprehensive results summary."""
    
    pd.set_option("display.float_format", "{:.4f}".format)
    
    print("\n" + "="*80)
    print("MULTI-CLASSIFIER COMPARISON (One-vs-Rest)")
    print("="*80)
    
    classifiers = list(results['metrics'].keys())
    
    # ========================================================================
    # OVERALL METRICS
    # ========================================================================
    print("\n" + "-"*80)
    print("OVERALL METRICS")
    print("-"*80)
    
    overall_metrics = {
        'Macro-AUC': 'Macro-AUC',
        'cmAP': 'cmAP',
        'PR-AUC (macro)': 'PR-AUC (macro)',
        'Brier (macro)': 'Brier (macro)',
        'Log-Loss (macro)': 'Log-Loss (macro)',
        'F1 (macro)': 'F1 (macro)',
        'Subset Accuracy': 'Subset Accuracy',
    }
    
    summary_data = []
    for clf_name in classifiers:
        metrics = results['metrics_per_fold'][clf_name]
        row = {'Classifier': clf_name}
        for key in overall_metrics.keys():
            row[key] = metrics.get(key, np.nan)
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data).set_index('Classifier')
    print(summary_df.to_string())
    
    # ========================================================================
    # PER-LABEL METRICS
    # ========================================================================
    print("\n" + "-"*80)
    print("PER-LABEL METRICS")
    print("-"*80)
    
    for metric_name in ['AP per label', 'PR-AUC per label', 'Brier per label', 'Log-Loss per label']:
        print(f"\n{metric_name}:")
        metric_data = []
        
        for clf_name in classifiers:
            metrics = results['metrics_per_fold'][clf_name]
            row = {'Classifier': clf_name}
            for label_name in label_names:
                row[label_name] = metrics[metric_name].get(label_name, np.nan)
            metric_data.append(row)
        
        metric_df = pd.DataFrame(metric_data).set_index('Classifier')
        print(metric_df.to_string())
    
    return summary_df


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results_comparison(results, label_names, figsize=(18, 12)):
    """Create comprehensive visualization of results."""
    
    classifiers = list(results['metrics'].keys())
    
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    
    # ========================================================================
    # ROW 1: Main metrics
    # ========================================================================
    
    # Macro-AUC
    ax = axes[0, 0]
    values = [results['metrics'][clf]['Macro-AUC'] for clf in classifiers]
    bars = ax.bar(classifiers, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'], 
                   edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Macro-AUC')
    ax.set_title('Macro-AUC (higher is better)', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # cmAP
    ax = axes[0, 1]
    values = [results['metrics'][clf]['cmAP'] for clf in classifiers]
    bars = ax.bar(classifiers, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                   edgecolor='black', linewidth=1.5)
    ax.set_ylabel('cmAP')
    ax.set_title('cmAP (higher is better)', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # PR-AUC
    ax = axes[0, 2]
    values = [results['metrics'][clf]['PR-AUC (macro)'] for clf in classifiers]
    bars = ax.bar(classifiers, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                   edgecolor='black', linewidth=1.5)
    ax.set_ylabel('PR-AUC')
    ax.set_title('PR-AUC Macro (higher is better)', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # ========================================================================
    # ROW 2: Error metrics
    # ========================================================================
    
    # Brier Score
    ax = axes[1, 0]
    values = [results['metrics'][clf]['Brier (macro)'] for clf in classifiers]
    bars = ax.bar(classifiers, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                   edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Brier Score')
    ax.set_title('Brier Score Macro (lower is better)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Log-Loss
    ax = axes[1, 1]
    values = [results['metrics'][clf]['Log-Loss (macro)'] for clf in classifiers]
    bars = ax.bar(classifiers, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                   edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Log-Loss')
    ax.set_title('Log-Loss Macro (lower is better)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # F1 Score
    ax = axes[1, 2]
    values = [results['metrics'][clf]['F1 (macro)'] for clf in classifiers]
    bars = ax.bar(classifiers, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                   edgecolor='black', linewidth=1.5)
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Macro (higher is better)', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # ========================================================================
    # ROW 3: Per-label heatmaps
    # ========================================================================
    
    for col_idx, metric_name in enumerate(['AP per label', 'PR-AUC per label', 'Brier per label']):
        ax = axes[2, col_idx]
        
        # Build matrix: classifiers × labels
        data_matrix = []
        for clf_name in classifiers:
            row = [results['metrics'][clf_name][metric_name][label] for label in label_names]
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # Heatmap
        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(label_names)))
        ax.set_xticklabels(label_names, rotation=45, ha='right')
        ax.set_yticks(range(len(classifiers)))
        ax.set_yticklabels(classifiers)
        ax.set_title(f'{metric_name}', fontweight='bold')
        
        # Add values
        for i in range(len(classifiers)):
            for j in range(len(label_names)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    fig.suptitle('Multi-Classifier Comparison (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("Multi-Classifier comparison module loaded.")
    print("\nUsage:")
    print("""
    from classifier_comparison import compare_classifiers, print_results_summary, plot_results_comparison
    
    # Load your data
    X = np.load("X_features2_not_normalized.npy")
    y = np.load("Y_labels2_not_normalized.npy")
    label_names = ['Type A', 'Type B', 'Type C', 'Type D', 'Echo']
    
    # Train and compare
    results = compare_classifiers_cv(X, y, label_names, test_size=0.2)
    
    # Print results
    summary_df = print_results_summary(results, label_names)
    
    # Plot results
    fig = plot_results_comparison(results, label_names)
    plt.savefig("classifier_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    """)

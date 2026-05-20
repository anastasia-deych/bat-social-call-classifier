import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc,
    log_loss,
    brier_score_loss,
    balanced_accuracy_score,
    hamming_loss,
    accuracy_score,
    f1_score,
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

def summarize(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return {
            "mean": values[0],
            "std": 0.0,
            "min": values[0],
            "max": values[0],
        }
    return {
        "mean": np.mean(values),
        "std": np.std(values, ddof=1),
        "min": np.min(values),
        "max": np.max(values),
    }

def compute_metrics(y_true, y_pred_proba, 
                    label_names=None, threshold=0.5):
    """Compute comprehensive multi-label metrics for a single evaluation."""
    y_pred_binary = (y_pred_proba > threshold).astype(int)
    
    n_labels = y_true.shape[1]
    if label_names is None:
        label_names = [f"Label {i}" for i in range(n_labels)]
    
    metrics = {
        'Macro-AUC': roc_auc_score(y_true, y_pred_proba, average='macro'),
        'cmAP': average_precision_score(y_true, y_pred_proba, average='macro'),
        #'Macro-Balanced Accuracy' : np.mean([balanced_accuracy_score(y_true[:, i], y_pred_binary[:, i]) for i in range(n_labels)]),
        'AP per label': {},
        #'Brier per label': {},
        #'Log-Loss per label': {}
    }
    
    #Per-label metrics
    for i, label_name in enumerate(label_names):
        # Average Precision (Recommended over trapezoidal PR-AUC)
        ap = average_precision_score(y_true[:, i], y_pred_proba[:, i])
        metrics['AP per label'][label_name] = ap
        
        # Error Scores
        #metrics['Brier per label'][label_name] = brier_score_loss(y_true[:, i], y_pred_proba[:, i])
        #metrics['Log-Loss per label'][label_name] = log_loss(y_true[:, i], y_pred_proba[:, i], labels=[0,1])
    
    # Aggregate macro scores
    metrics['Brier (macro)'] = np.mean( [brier_score_loss(y_true[:, i], y_pred_proba[:, i]) for i in range(n_labels)] )
    metrics['Log-Loss (macro)'] = np.mean( [log_loss(y_true[:, i], y_pred_proba[:, i], labels=[0,1]) for i in range(n_labels)] )
    
    return metrics

def compute_fold_metrics(y_true, y_pred_proba, label_names=None, threshold=0.5):
    fold_metrics = []
    for fold in range(len(y_true)):
        fold_metrics.append(compute_metrics(
            y_true[fold], 
            y_pred_proba[fold], 
            label_names, 
            threshold
        ))
    return fold_metrics


def compute_cv_stats(fold_metrics):
    """Compute metrics across cross-validation folds."""
    result = {}
    metrics = fold_metrics[0].keys()
    for metric in metrics:

        first_value = fold_metrics[0][metric]

        # scalar metric
        if np.isscalar(first_value):

            vals = [m[metric] for m in fold_metrics]
            result[metric] = summarize(vals)

        # nested dict metric
        elif isinstance(first_value, dict):

            result[metric] = {}

            labels = first_value.keys()

            for label in labels:
                vals = [m[metric][label] for m in fold_metrics]
                result[metric][label] = summarize(vals)

        else:
            raise TypeError(f"Unsupported metric type for key={metric}")
    
    return result

def result_summary(y_true, y_pred_proba, label_names=None, threshold=0.5,stats : bool = True) :
    oof_true = np.concatenate(y_true, axis=0)
    oof_pred_proba = np.concatenate(y_pred_proba, axis=0)
    fold_metrics = compute_fold_metrics(y_true, y_pred_proba, label_names, threshold)

    if stats :
        results = compute_cv_stats(fold_metrics)
    else :
        results = {
            "oof" :{
                "metrics" : compute_metrics(oof_true, oof_pred_proba, label_names, threshold),
                "true": oof_true,
                "pred_proba": oof_pred_proba
            },
            "cv" : {
                "stats" : compute_cv_stats(fold_metrics),
                "folds": fold_metrics
            }
        }
    return results

def compile_results(all_results,label_names=None,stats : bool = True,encoder : str = 'perch2') :
    if label_names is None:
        label_names = ['Type A', 'Type B', 'Type C', 'Type D', 'Echo']
    
    compiled_results = {}
    unique_models = list(set([r['model'] for r in all_results]))
    for model_name in unique_models:
        # 1. Gather all trials for this specific classifier
        model_trials = [r for r in all_results if r['model'] == model_name]
        y_true = []
        y_pred_proba = []
        for trial_data in model_trials:
            # These are already concatenated across the outer folds!
            y_true.append(trial_data['oof_y_true'])
            y_pred_proba.append(trial_data['oof_y_pred_proba'])
        # 2. Compute summary metrics for this model
        model_results = result_summary(y_true, y_pred_proba, label_names, stats=stats)
        compiled_results[encoder + " " + model_name] = model_results

    return compiled_results

def generate_metrics_table2(all_results,label_names=None) :
    global_rows = []
    class_rows = []

    # Dynamic class names extracted safely from your labels key
    if label_names is None:
        class_names = ['Type A', 'Type B', 'Type C', 'Type D', 'Echo']
    else:
        class_names = label_names

    for model, stats in all_results.items():
        # 1. Parse Global Data Matrix
        global_row = {
            "Model": model,
            "Macro-AUC": f"{stats['Macro-AUC']['mean']:.3f} ± {stats['Macro-AUC']['std']:.3f}",
            "Macro-AP (cmAP)": f"{stats['cmAP']['mean']:.3f} ± {stats['cmAP']['std']:.3f}",
            #"Balanced Accuracy": f"{stats['Macro-Balanced Accuracy']['mean']:.3f} ± {stats['Macro-Balanced Accuracy']['std']:.3f}",
            "Brier Score ↓": f"{stats['Brier (macro)']['mean']:.3f} ± {stats['Brier (macro)']['std']:.3f}",
            "Log-Loss ↓": f"{stats['Log-Loss (macro)']['mean']:.3f} ± {stats['Log-Loss (macro)']['std']:.3f}"
        }
        global_rows.append(global_row)

        # 2. Parse Class-Specific Average Precision Data Matrix
        class_row = {"Model": model}
        for label in class_names:
            label_stats = stats['AP per label'][label]
            class_row[f"{label} AP"] = f"{label_stats['mean']:.3f} ± {label_stats['std']:.3f}"
        class_rows.append(class_row)

    # Convert arrays into clean Pandas DataFrames
    global_df = pd.DataFrame(global_rows)
    class_df = pd.DataFrame(class_rows)
    return global_df, class_df

def generate_metrics_table(all_results,label_names=None):
    global_rows = []
    class_rows = []
    for model, stats in all_results.items():
        global_row = {
            "Model": model,
            "Macro-AUC": f"{stats['Macro-AUC'][0]:.3f} ± {max(stats['Macro-AUC'][1]-stats['Macro-AUC'][0], stats['Macro-AUC'][0]-stats['Macro-AUC'][2]) :.3f}",
            "cmAP": f"{stats['cmAP'][0]:.3f} ± {max(stats['cmAP'][1]-stats['cmAP'][0], stats['cmAP'][0]-stats['cmAP'][2]):.3f}",
            "Brier Score": f"{stats['Brier (macro) mean'][0]:.4f} ± {max(stats['Brier (macro) mean'][1]-stats['Brier (macro) mean'][0], stats['Brier (macro) mean'][0]-stats['Brier (macro) mean'][2]):.3f}",
            "Log-Loss": f"{stats['Log-Loss (macro) mean'][0]:.3f} ± {max(stats['Log-Loss (macro) mean'][1]-stats['Log-Loss (macro) mean'][0], stats['Log-Loss (macro) mean'][0]-stats['Log-Loss (macro) mean'][2]):.3f}"
        }
        global_rows.append(global_row)

        class_row = {
            "Model": model,
            "AP type A": f"{stats['AP per label'][0][label_names[0]]:.3f} ± {max(stats['AP per label'][1][label_names[0]]-stats['AP per label'][0][label_names[0]], stats['AP per label'][0][label_names[0]]-stats['AP per label'][2][label_names[0]]):.3f}",
            "AP type B": f"{stats['AP per label'][0][label_names[1]]:.3f} ± {max(stats['AP per label'][1][label_names[1]]-stats['AP per label'][0][label_names[1]], stats['AP per label'][0][label_names[1]]-stats['AP per label'][2][label_names[1]]):.3f}",
            "AP type C": f"{stats['AP per label'][0][label_names[2]]:.3f} ± {max(stats['AP per label'][1][label_names[2]]-stats['AP per label'][0][label_names[2]], stats['AP per label'][0][label_names[2]]-stats['AP per label'][2][label_names[2]]):.3f}",
            "AP type D": f"{stats['AP per label'][0][label_names[3]]:.3f} ± {max(stats['AP per label'][1][label_names[3]]-stats['AP per label'][0][label_names[3]], stats['AP per label'][0][label_names[3]]-stats['AP per label'][2][label_names[3]]):.3f}",
            "AP Echo": f"{stats['AP per label'][0][label_names[4]]:.3f} ± {max(stats['AP per label'][1][label_names[4]]-stats['AP per label'][0][label_names[4]], stats['AP per label'][0][label_names[4]]-stats['AP per label'][2][label_names[4]]):.3f}"
        }
        class_rows.append(class_row)

    global_df = pd.DataFrame(global_rows)
    class_df = pd.DataFrame(class_rows)
    return global_df, class_df

"""Example Usage in Notebook
df_results = generate_metrics_table(results_vault)
display(df_results)
"""

def plot_model_comparison(all_results, metrics_to_plot=None, title="Model Performance Comparison"):
    """
    Plots mean performance with [Min, Max] error bars for multiple models.
    
    Args:
        all_results (dict): Dictionary where keys are model names and values are 
                            the output of your compute_cv_stats function.
        metrics_to_plot (list): List of metric keys to include (e.g., ['Macro-AUC', 'cmAP'])
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['Macro-AUC', 'cmAP', 'Brier (macro) mean', 'Log-Loss (macro) mean']
    
    models = list(all_results.keys())
    n_metrics = len(metrics_to_plot)
    n_models = len(models)
    
    # Set up the plot dimensions
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Grouped bar settings
    width = 0.8 / n_models  # Total group width is 0.8
    x = np.arange(n_metrics)
    
    # Standard colors for bats/nature themes
    colors = plt.cm.viridis(np.linspace(0, 0.8, n_models))

    for i, model_name in enumerate(models):
        means = []
        lower_err = []
        upper_err = []
        
        for m_key in metrics_to_plot:
            stats = all_results[model_name][m_key] # [mean, max, min]
            
            mean_val = stats[0]
            max_val = stats[1]
            min_val = stats[2]
            
            means.append(mean_val)
            # Matplotlib yerr format: [ [lower_offsets], [upper_offsets] ]
            lower_err.append(mean_val - min_val)
            upper_err.append(max_val - mean_val)
        
        # Calculate x-offset for this specific model's bars
        offset = i * width - (width * n_models) / 2 + width / 2
        
        ax.bar(x + offset, means, width, 
               yerr=[lower_err, upper_err], 
               label=model_name, 
               color=colors[i],
               capsize=5, 
               alpha=0.85,
               edgecolor='white')

    # Formatting
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, fontsize=11)
    ax.set_ylabel("Score (Mean with Min/Max Range)")
    ax.legend(title="Algorithms", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_comprehensive_results(all_results, labels, title="Model Evaluation"):
    sns.set_context("paper")
    sns.set_style("whitegrid")
    
    # We now have 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(all_results)))
    width = 0.8 / len(all_results)

    # --- Subplot 1: Global Metrics (Macro-AUC & cmAP) ---
    ax1 = axes[0]
    global_keys = ['Macro-AUC', 'cmAP']
    x_global = np.arange(len(global_keys))

    for i, (model_name, stats) in enumerate(all_results.items()):
        means = [stats[k][0] for k in global_keys]
        yerr = [[stats[k][0]-stats[k][2] for k in global_keys], 
                [stats[k][1]-stats[k][0] for k in global_keys]]
        
        offset = i * width - (width * len(all_results)) / 2 + width / 2
        ax1.bar(x_global + offset, means, width, yerr=yerr, label=model_name, 
                color=colors[i], capsize=5, alpha=0.8, edgecolor='white')

    ax1.set_title("Global Performance: Macro-AUC vs cmAP", fontsize=14, fontweight='bold')
    ax1.set_xticks(x_global)
    ax1.set_xticklabels(['Macro-AUC', 'cmAP (mAP)'], fontsize=12)
    ax1.set_ylabel("Score (0.0 - 1.0)")
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='upper right')

    # --- Subplot 2: Per-Class Average Precision ---
    ax2 = axes[1]
    x_labels = np.arange(len(labels))
    
    for i, (model_name, stats) in enumerate(all_results.items()):
        ap_data = stats['AP per label'] # [mean_dict, max_dict, min_dict]
        means = [ap_data[0][l] for l in labels]
        yerr = [[ap_data[0][l] - ap_data[2][l] for l in labels], 
                [ap_data[1][l] - ap_data[0][l] for l in labels]]
        
        offset = i * width - (width * len(all_results)) / 2 + width / 2
        ax2.bar(x_labels + offset, means, width, yerr=yerr, 
                color=colors[i], capsize=3, alpha=0.8, edgecolor='white')

    ax2.set_title("Per-Class Average Precision", fontsize=14, fontweight='bold')
    ax2.set_xticks(x_labels)
    ax2.set_xticklabels(labels, rotation=35, ha='right')
    ax2.set_ylabel("AP Score")
    ax2.set_ylim(0, 1.1)

    # --- Subplot 3: Error Metrics (Brier & Log-Loss) ---
    ax3 = axes[2]
    error_metrics = ['Brier (macro) mean', 'Log-Loss (macro) mean']
    x_err = np.arange(len(error_metrics))
    
    for i, (model_name, stats) in enumerate(all_results.items()):
        means = [stats[m][0] for m in error_metrics]
        yerr = [[stats[m][0]-stats[m][2] for m in error_metrics], 
                [stats[m][1]-stats[m][0] for m in error_metrics]]
        
        offset = i * width - (width * len(all_results)) / 2 + width / 2
        ax3.bar(x_err + offset, means, width, yerr=yerr, 
                color=colors[i], capsize=5, alpha=0.8, edgecolor='white')

    ax3.set_title("Calibration Error", fontsize=14, fontweight='bold')
    ax3.set_xticks(x_err)
    ax3.set_xticklabels(['Brier Score', 'Log-Loss'], fontsize=12)
    ax3.set_ylabel("Error Value")

    plt.suptitle(title, fontsize=20, y=1.02)
    plt.tight_layout()
    plt.show()



def plot_comprehensive_results2(all_results, labels, title="Model Evaluation"):
    sns.set_context("paper")
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 20)) # Increased size slightly
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(all_results)))
    width = 0.8 / len(all_results)
    
    handles, legend_labels = [], []

    # --- HELPER: Manual Zoom Function ---
    def apply_manual_zoom(ax, data_points, padding=0.1):
        """Forces the Y-axis to zoom in on the data range."""
        if not data_points: return
        d_min, d_max = min(data_points), max(data_points)
        diff = d_max - d_min
        # If there's no difference (e.g. all 1.0), use a default range
        if diff == 0:
            ax.set_ylim(d_min - 0.05, d_min + 0.05)
        else:
            ax.set_ylim(d_min - (diff * padding), d_max + (diff * padding))

    # --- Subplot 1: Global Metrics ---
    ax1 = axes[0]
    global_keys = ['Macro-AUC', 'cmAP']
    x_global = np.arange(len(global_keys))
    points_for_zoom1 = []

    for i, (model_name, stats) in enumerate(all_results.items()):
        means = [stats[k][0] for k in global_keys]
        low_err = [stats[k][0] - stats[k][2] for k in global_keys]
        high_err = [stats[k][1] - stats[k][0] for k in global_keys]
        
        # Track every low/high point for scaling
        points_for_zoom1.extend([m - l for m, l in zip(means, low_err)])
        points_for_zoom1.extend([m + h for m, h in zip(means, high_err)])
        
        offset = i * width - (width * len(all_results)) / 2 + width / 2
        bar = ax1.bar(x_global + offset, means, width, yerr=[low_err, high_err], 
                      color=colors[i], capsize=4, alpha=0.8, edgecolor='white')
        
        if model_name not in legend_labels:
            handles.append(bar)
            legend_labels.append(model_name)

    ax1.set_title("Global Performance: Macro-AUC vs cmAP", fontsize=15, fontweight='bold')
    ax1.set_xticks(x_global)
    ax1.set_xticklabels(['Macro-AUC', 'cmAP (mAP)'], fontsize=12)
    ax1.set_ylabel("Score")
    apply_manual_zoom(ax1, points_for_zoom1)

    # --- Subplot 2: Per-Class Average Precision ---
    ax2 = axes[1]
    x_labels = np.arange(len(labels))
    points_for_zoom2 = []
    
    for i, (model_name, stats) in enumerate(all_results.items()):
        ap_data = stats['AP per label']
        means = [ap_data[0][l] for l in labels]
        low_err = [ap_data[0][l] - ap_data[2][l] for l in labels]
        high_err = [ap_data[1][l] - ap_data[0][l] for l in labels]
        
        points_for_zoom2.extend([m - l for m, l in zip(means, low_err)])
        points_for_zoom2.extend([m + h for m, h in zip(means, high_err)])
        
        offset = i * width - (width * len(all_results)) / 2 + width / 2
        ax2.bar(x_labels + offset, means, width, yerr=[low_err, high_err], 
                color=colors[i], capsize=3, alpha=0.8, edgecolor='white')

    ax2.set_title("Per-Class Average Precision", fontsize=15, fontweight='bold')
    ax2.set_xticks(x_labels)
    ax2.set_xticklabels(labels, rotation=35, ha='right')
    ax2.set_ylabel("AP Score")
    apply_manual_zoom(ax2, points_for_zoom2)

    # --- Subplot 3: Error Metrics ---
    ax3 = axes[2]
    error_metrics = ['Brier (macro) mean', 'Log-Loss (macro) mean']
    x_err = np.arange(len(error_metrics))
    points_for_zoom3 = []
    
    for i, (model_name, stats) in enumerate(all_results.items()):
        means = [stats[m][0] for m in error_metrics]
        low_err = [stats[m][0] - stats[m][2] for m in error_metrics]
        high_err = [stats[m][1] - stats[m][0] for m in error_metrics]
        
        points_for_zoom3.extend([m - l for m, l in zip(means, low_err)])
        points_for_zoom3.extend([m + h for m, h in zip(means, high_err)])
        
        offset = i * width - (width * len(all_results)) / 2 + width / 2
        ax3.bar(x_err + offset, means, width, yerr=[low_err, high_err], 
                color=colors[i], capsize=5, alpha=0.8, edgecolor='white')

    ax3.set_title("Calibration Error (Lower is Better)", fontsize=15, fontweight='bold')
    ax3.set_xticks(x_err)
    ax3.set_xticklabels(['Brier Score', 'Log-Loss'], fontsize=12)
    ax3.set_ylabel("Error Value")
    apply_manual_zoom(ax3, points_for_zoom3)

    # --- Global Legend and Layout ---
    plt.suptitle(title, fontsize=22, y=1.02)
    
    # Legend at bottom with multiple rows if needed
    fig.legend(handles, legend_labels, loc='lower center', ncol=3, 
               bbox_to_anchor=(0.5, -0.05), fontsize=11, frameon=True)

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.show()

def plot_comprehensive_results3(all_results, labels, title="Model Evaluation"):
    sns.set_context("paper")
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 20)) 
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(all_results)))
    width = 0.8 / len(all_results)
    
    handles, legend_labels = [], []

    # --- HELPER: Manual Zoom Function ---
    def apply_manual_zoom(ax, data_points, padding=0.15):
        """Forces the Y-axis to zoom in on the data range."""
        if not data_points: return
        d_min, d_max = min(data_points), max(data_points)
        diff = d_max - d_min
        if diff == 0:
            ax.set_ylim(max(0, d_min - 0.05), min(1.0, d_min + 0.05))
        else:
            # Set boundaries gracefully, ensuring we don't zoom out past logical limits (like 0)
            ax.set_ylim(max(0, d_min - (diff * padding)), min(d_max + (diff * padding), d_max * 1.5))

    # --- Subplot 1: Global Metrics ---
    ax1 = axes[0]
    global_keys = ['Macro-AUC', 'cmAP']
    x_global = np.arange(len(global_keys))
    points_for_zoom1 = []

    for i, (model_name, stats) in enumerate(all_results.items()):
        # Extract values using standard string keys instead of positional integers
        means = [stats[k]['mean'] for k in global_keys]
        stds = [stats[k]['std'] for k in global_keys]
        
        # Track data bounds for the visual zoom
        for m, s in zip(means, stds):
            points_for_zoom1.extend([m - s, m + s])
        
        offset = i * width - (width * len(all_results)) / 2 + width / 2
        bar = ax1.bar(x_global + offset, means, width, yerr=stds, 
                      color=colors[i], capsize=5, alpha=0.8, edgecolor='white')
        
        if model_name not in legend_labels:
            handles.append(bar)
            legend_labels.append(model_name)

    ax1.set_title("Global Performance: Macro-AUC vs cmAP", fontsize=15, fontweight='bold')
    ax1.set_xticks(x_global)
    ax1.set_xticklabels(['Macro-AUC', 'cmAP (mAP)'], fontsize=12)
    ax1.set_ylabel("Score")
    apply_manual_zoom(ax1, points_for_zoom1)

    # --- Subplot 2: Per-Class Average Precision ---
    ax2 = axes[1]
    x_labels = np.arange(len(labels))
    points_for_zoom2 = []
    
    for i, (model_name, stats) in enumerate(all_results.items()):
        ap_data = stats['AP per label']
        
        # Pull out mean and std for each specific target class label
        means = [ap_data[l]['mean'] for l in labels]
        stds = [ap_data[l]['std'] for l in labels]
        
        for m, s in zip(means, stds):
            points_for_zoom2.extend([m - s, m + s])
        
        offset = i * width - (width * len(all_results)) / 2 + width / 2
        ax2.bar(x_labels + offset, means, width, yerr=stds, 
                color=colors[i], capsize=4, alpha=0.8, edgecolor='white')

    ax2.set_title("Per-Class Average Precision", fontsize=15, fontweight='bold')
    ax2.set_xticks(x_labels)
    ax2.set_xticklabels(labels, rotation=35, ha='right', fontsize=11)
    ax2.set_ylabel("AP Score")
    apply_manual_zoom(ax2, points_for_zoom2)

    # --- Subplot 3: Error Metrics ---
    ax3 = axes[2]
    # Fixed keys to perfectly match the exact strings inside your dataset
    error_metrics = ['Brier (macro)', 'Log-Loss (macro)']
    x_err = np.arange(len(error_metrics))
    points_for_zoom3 = []
    
    for i, (model_name, stats) in enumerate(all_results.items()):
        means = [stats[m]['mean'] for m in error_metrics]
        stds = [stats[m]['std'] for m in error_metrics]
        
        for m, s in zip(means, stds):
            points_for_zoom3.extend([m - s, m + s])
        
        offset = i * width - (width * len(all_results)) / 2 + width / 2
        ax3.bar(x_err + offset, means, width, yerr=stds, 
                color=colors[i], capsize=5, alpha=0.8, edgecolor='white')

    ax3.set_title("Calibration Error (Lower is Better)", fontsize=15, fontweight='bold')
    ax3.set_xticks(x_err)
    ax3.set_xticklabels(['Brier Score', 'Log-Loss'], fontsize=12)
    ax3.set_ylabel("Error Value")
    apply_manual_zoom(ax3, points_for_zoom3)

    # --- Global Legend and Layout Adjustments ---
    plt.suptitle(title, fontsize=22, y=1.01, fontweight='bold')
    
    fig.legend(handles, legend_labels, loc='lower center', ncol=3, 
               bbox_to_anchor=(0.5, -0.02), fontsize=12, frameon=True)

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.show()



"""Implementation
# 1. Collect your stats into a dictionary
labels = ['Type A', 'Type B', 'Type C', 'Type D', 'Echo']
results_vault = {
    "Perch 2.0 SVM": compute_cv_stats(y_true_perch_svm, y_prob_perch_svm, label_names=labels),
    "Perch 2.0 RF": compute_cv_stats(y_true_perch_rf, y_prob_perch_rf, label_names=labels),
    "Perch 2.0 MLP": compute_cv_stats(y_true_perch_mlp, y_prob_perch_mlp, label_names=labels),
    "NLM BEATs": compute_cv_stats(y_true_beats, y_prob_beats, label_names=labels),
    "EffNet B0": compute_cv_stats(y_true_eff, y_prob_eff, label_names=labels)
    }

# 2. Call the plot
plot_model_comparison(
    all_results=results_vault, 
    metrics_to_plot=['Macro-AUC', 'cmAP'], # Choose which metrics to show
    title="Pipistrelle Classification: Encoder Comparison"
)
"""

def plot_calibration_curves(y_true,y_pred_proba,label_names=None,n_bins=10,strategy="quantile"):
    """
    Plot calibration curves + probability histograms
    for multilabel classification.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples, n_labels)
        Binary ground-truth matrix.

    y_pred_proba : ndarray of shape (n_samples, n_labels)
        Predicted probabilities.

    label_names : list[str], optional
        Names of labels.

    n_bins : int
        Number of calibration bins.

    strategy : {"uniform", "quantile"}
        Binning strategy for calibration_curve.
    """
    n_labels = y_true.shape[1]

    if label_names is None:
        label_names = [f"Label {i}" for i in range(n_labels)]

    # 2 rows: top calibration curves, bottom histograms
    fig, axes = plt.subplots(2,n_labels,figsize=(5 * n_labels, 8))

    # handle case n_labels == 1
    if n_labels == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i in range(n_labels):
        # Skip degenerate labels
        if len(np.unique(y_true[:, i])) < 2:
            axes[0, i].set_visible(False)
            axes[1, i].set_visible(False)
            continue
        
        brier = brier_score_loss(y_true[:, i],y_pred_proba[:, i])
        # ---Calibration curve---------------
        prob_true, prob_pred = calibration_curve(y_true[:, i],y_pred_proba[:, i],n_bins=n_bins,strategy=strategy)

        ax_curve = axes[0, i]
        ax_curve.plot(prob_pred,prob_true,marker='o',linewidth=2)
        # perfect calibration
        ax_curve.plot([0, 1],[0, 1],linestyle='--',color='gray')

        ax_curve.set_title(f"{label_names[i]}\nBrier={brier:.3f}")
        ax_curve.set_xlabel("Mean predicted probability")
        ax_curve.set_ylabel("Fraction of positives")
        ax_curve.set_xlim(0, 1)
        ax_curve.set_ylim(0, 1)
        ax_curve.grid(True)

        # ---Histogram-----------------------
        ax_hist = axes[1, i]
        ax_hist.hist(y_pred_proba[:, i],bins=n_bins,alpha=0.7)

        ax_hist.set_title(f"{label_names[i]} Probability Distribution")
        ax_hist.set_xlabel("Predicted probability")
        ax_hist.set_ylabel("Count")
        ax_hist.set_xlim(0, 1)
        ax_hist.grid(True)

    plt.tight_layout()
    plt.show()

def plot_comprehensive_calibration(encoder_results, label_names, n_bins=10, strategy="uniform"):
    """
    Plots a multi-model, multi-encoder calibration comparison grid.
    Each column represents a target label (Class).
    
    Parameters
    ----------
    encoder_results : dict
        A nested dictionary structured as:
        {
            "EffNet": [list of trial dicts containing 'model', 'oof_y_true', 'oof_y_pred_proba'],
            "Perch 2.0": [list of trial dicts ...]
        }
    label_names : list[str]
        Names of the target classes (e.g., ['Type A', 'Type B', 'Type C', 'Type D', 'Echolocation'])
    """
    sns.set_context("paper")
    sns.set_style("whitegrid")
    
    n_labels = len(label_names)
    
    # 1 Row for Calibration Curves, 1 Row for Histograms
    fig, axes = plt.subplots(2, n_labels, figsize=(4.5 * n_labels, 9))
    
    # Generate distinct, beautiful color palettes dynamically
    all_combinations = []
    for encoder_name in encoder_results.keys():
        # Look inside the first trial to check available classifier models
        models = sorted(list(set([r['model'] for r in encoder_results[encoder_name]])))
        for model in models:
            all_combinations.append((encoder_name, model))
            
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_combinations)))
    combo_colors = {combo: colors[idx] for idx, combo in enumerate(all_combinations)}
    
    legend_handles = {}

    # Iterate through each column (Each target class label)
    for class_idx, label_name in enumerate(label_names):
        ax_curve = axes[0, class_idx]
        ax_hist = axes[1, class_idx]
        
        # Perfect calibration reference line
        ax_curve.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7, label='Perfect Calibration')
        
        for encoder_name, trial_list in encoder_results.items():
            models = sorted(list(set([r['model'] for r in trial_list])))
            
            for model_name in models:
                # Gather all independent trials matching this specific combination
                model_trials = [r for r in trial_list if r['model'] == model_name]
                
                trial_prob_true = []
                # Use standard uniform bin coordinates to ensure clean alignment during averaging
                common_bins = np.linspace(0, 1, n_bins)
                bin_centers = (common_bins[:-1] + common_bins[1:]) / 2
                
                # Accumulator for historical distribution checks
                all_pred_probas = []
                
                for trial in model_trials:
                    y_true = trial['oof_y_true'][:, class_idx]
                    y_prob = trial['oof_y_pred_proba'][:, class_idx]
                    all_pred_probas.extend(y_prob)
                    
                    # Calculate structural curve coordinates for this specific trial
                    p_true, p_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
                    
                    # Interp forces coordinates to map cleanly onto a shared x-axis layout for averaging
                    interp_true = np.interp(bin_centers, p_pred, p_true, left=np.nan, right=np.nan)
                    trial_prob_true.append(interp_true)
                
                # Average across Dimension 3 (Trials) safely ignoring any empty boundary bins
                mean_prob_true = np.nanmean(trial_prob_true, axis=0)
                
                # Plot setup
                combo_key = (encoder_name, model_name)
                current_color = combo_colors[combo_key]
                display_label = f"{encoder_name} - {model_name}"
                
                # --- Plot Calibration Curves ---
                # Mask out NaN bins where no predictions landed during testing
                valid_mask = ~np.isnan(mean_prob_true)
                line, = ax_curve.plot(bin_centers[valid_mask], mean_prob_true[valid_mask], 
                                      marker='o', markersize=4, linewidth=2, 
                                      color=current_color, alpha=0.85)
                
                if display_label not in legend_handles:
                    legend_handles[display_label] = line
                
                # --- Plot Density Histograms (Step/Outline styles keep multi-lines legible) ---
                ax_hist.hist(all_pred_probas, bins=common_bins, histtype='step', 
                             linewidth=1.5, color=current_color, alpha=0.75)
        
        # Formatting Top Subplot Row
        ax_curve.set_title(f"Calibration: {label_name}", fontsize=14, fontweight='bold')
        ax_curve.set_xlabel("Mean Predicted Probability", fontsize=10)
        ax_curve.set_ylabel("Fraction of Positives", fontsize=10)
        ax_curve.set_xlim(0, 1)
        ax_curve.set_ylim(0, 1)
        
        # Formatting Bottom Subplot Row
        ax_hist.set_title(f"Distribution: {label_name}", fontsize=14, fontweight='bold')
        ax_hist.set_xlabel("Predicted Probability", fontsize=10)
        ax_hist.set_ylabel("Density / Sample Count", fontsize=10)
        ax_hist.set_xlim(0, 1)
        ax_hist.set_yscale('log') # Log scale helps check minor boundaries when predictions stack at 0 or 1

    # Place a single unified legend block neatly below the chart grid
    fig.legend(legend_handles.values(), legend_handles.keys(), loc='lower center', 
               ncol=min(4, len(legend_handles)), bbox_to_anchor=(0.5, -0.06), 
               fontsize=11, frameon=True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()


def label_confusion(y_true,y_pred_proba,y_pred_binary=None, label_names=None, threshold=0.5) :
    """
    Analyzes which labels are predicted 'instead' of the true labels.
    """
    # 1. Convert proba to binary if binary isn't provided
    if y_pred_binary is None:
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    y_true = np.array(y_true)
    y_pred_binary = np.array(y_pred_binary)
    num_labels = y_true.shape[1]
    
    if label_names is None:
        label_names = [f"Label_{i}" for i in range(num_labels)]

    # 2. Initialize Confusion Matrix
    # Rows: The label that was SHOULD have been there (False Negative)
    # Cols: The label that was predicted WRONGLY (False Positive)
    confusion_mtx = np.zeros((num_labels, num_labels))

    # 3. Iterate through samples
    for i in range(len(y_true)):
        actual = y_true[i]
        pred = y_pred_binary[i]

        # Indices of missed labels (FN)
        missed = np.where((actual == 1) & (pred == 0))[0]
        # Indices of extra labels (FP)
        extra = np.where((actual == 0) & (pred == 1))[0]

        # If we missed something AND predicted something else wrongly
        for m_idx in missed:
            for e_idx in extra:
                confusion_mtx[m_idx, e_idx] += 1

    # 4. Wrap in DataFrame for easy viewing
    df_cm = pd.DataFrame(confusion_mtx, index=label_names, columns=label_names)
    
    return df_cm



import numpy as np
from preprocessing.dataset import AugmentationPipeline
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def get_balanced_split_indices(X, y, folds = 5, random_state=42):
    np.random.seed(random_state)
    
    # 1. First, split the data proportionally using iterative stratification
    n_splits = folds
    kf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    strat_train_idx, strat_test_idx = next(kf.split(X, y))
    
    # 2. Determine our target uniform count based on the stratified test set
    y_strat_test = y[strat_test_idx]
    original_counts = np.sum(y_strat_test, axis=0)
    
    # Set the uniform target to the average or median class size to keep it healthy
    # (Using the absolute minimum creates too small of a bottleneck)
    target_uniform_count = int(np.median(original_counts))
    
    print(f"Original stratified test counts per class: {original_counts}")
    print(f"Target uniform capacity per class: {target_uniform_count}")
    
    # 3. Calculate label "rarity" weights (inverse frequencies)
    # Rare labels get higher priority so they aren't accidentally discarded
    class_frequencies = np.sum(y, axis=0)
    label_weights = 1.0 / (class_frequencies + 1e-5)
    
    # Assign a priority score to each sample in the test set based on its rarest label
    sample_priorities = np.dot(y_strat_test, label_weights)
    
    # Sort test indices: rarest samples first
    sorted_test_meta_indices = np.argsort(sample_priorities)[::-1]
    sorted_actual_test_idxs = strat_test_idx[sorted_test_meta_indices]
    
    # 4. Filter test set with a capacity ceiling
    final_test_idx = []
    final_train_idx = list(strat_train_idx)
    
    current_test_counts = np.zeros(y.shape[1])
    
    for idx in sorted_actual_test_idxs:
        sample_labels = y[idx]
        active_classes = np.where(sample_labels == 1)[0]
        
        # Check if adding this sample would violate the capacity limit for ALL its active classes
        # If it fits in at least one under-represented class, we keep it!
        can_fit = any(current_test_counts[c] < target_uniform_count for c in active_classes)
        
        if can_fit or len(active_classes) == 0: # Always keep negative/empty samples if any
            final_test_idx.append(idx)
            current_test_counts += sample_labels
        else:
            # If it exceeds capacity, safely move it back to the training partition
            final_train_idx.append(idx)
            
    final_train_idx = np.array(final_train_idx)
    final_test_idx = np.array(final_test_idx)
    
    print(f"Final balanced test counts per class: {np.sum(y[final_test_idx], axis=0)}")
    print(f"Final split size: Train={len(final_train_idx)}, Test={len(final_test_idx)}")
    
    return final_train_idx, final_test_idx
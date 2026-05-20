



from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from models.linear_probes import BalancedMLP, MultilabelPrevalenceBaseline
from preprocessing.data_splitter import get_balanced_split_indices
import numpy as np

def balanced_test(X, y,folds , random_state=42,trials=5):
    # 1. Initialize the split
    clf_names = ['SVM', 'Random Forest', 'MLP','Random Guesser']

    # 2. Setup storage
    all_results = []

    models = {
        'SVM': OneVsRestClassifier(SVC(
            probability=True, 
            random_state=random_state)),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            random_state=random_state), # RF is natively multi-label
        
        #'Random Forest': OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=random_state)),
        #'MLP' : OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=random_state))
        
        'MLP': BalancedMLP(
            input_dim=X.shape[1],
            hidden_dim=128,
            lr=0.001,
            epochs=50,
            dropout=0.2,
            batch_norm=False
        ),
        'Random Guesser' : MultilabelPrevalenceBaseline(type='stochastic')
    }


    for i in range(trials) :
        for name, clf in models.items():
            train_idx, test_idx = get_balanced_split_indices(X, y, folds=folds, random_state=random_state+i)
            print(f"Trial {i+1}, Model: {name}, Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)


            clf.fit(X_train, y_train)
            y_proba = clf.predict_proba(X_test)

            # predict_proba for multi-label often returns a list of arrays
            # We want to ensure it's a consistent [Samples, Labels] array
            if isinstance(y_proba, list):
                # Convert list of [Samples, 2] to [Samples, Labels] using the positive class proba
                y_proba = np.array([p[:, 1] for p in y_proba]).T

            all_results.append({
                    'trial': i,
                    'model': name,
                    'oof_y_true': y_test,
                    'oof_y_pred_proba': y_proba,
                    'oof_indices': test_idx
                })

    
    # Return as numpy arrays for easier use in your compute_cv_stats
    return all_results
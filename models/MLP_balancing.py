



from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from models.linear_probes import BalancedMLP, MultilabelPrevalenceBaseline
from models.focal_loss import FocalLoss
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np

from preprocessing.dataset import AugmentationPipeline


def balancing_mlp(X, y, n_split_out=5,n_split_in=5, num_trials=5,random_state=42,balance : bool = False):
    # 1. Initialize the split
    scorer = make_scorer(average_precision_score, average='macro', response_method='predict_proba')
    all_results = []

    for i in range(num_trials) :
        print(f"Starting Trial {i+1}/{num_trials} with random_state={random_state + i}...")
        model_params = {
            'MLP_Baseline': {
                'model': BalancedMLP(input_dim=X.shape[1],balanced=False),
                'params': {} # Plain Binary Cross Entropy
            },
            'MLP_ClassWeights': {
                'model': BalancedMLP(input_dim=X.shape[1],balanced=True),
                'params': {} # Scaled BCE loss based on minority prevalence
            },
            'MLP_FocalLoss': {
                'model': BalancedMLP(input_dim=X.shape[1],focal_loss=True),
                'params': {
                    'model__focal_gamma': [1.0, 2.0, 5.0], # Higher gamma = focus more on hard bat calls
                    'model__focal_alpha': [0.25, 0.5, 0.75]
                }
            },
            'MLP_Oversampled': {
                # Use standard BCE but feed it data passed through an oversampler in your pipeline
                'model': BalancedMLP(input_dim=X.shape[1],balanced=False),
                'params': {}
            }
        }
        #Cross validation techniques for inner and outer loop
        inner_cv = MultilabelStratifiedKFold(n_splits=n_split_in, shuffle=True, random_state=i)
        outer_cv = MultilabelStratifiedKFold(n_splits=n_split_out, shuffle=True, random_state=i)

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

                if model_name == 'MLP_Oversampled' :
                    clf = mp['model']
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    X_train,y_train = AugmentationPipeline().iterative_oversample(X_train,y_train,random_state=42+i)
                    clf.fit(X_train, y_train)
                else :
                    clf = GridSearchCV(estimator=pipeline,param_grid=mp['params'],cv=inner_cv,
                                   scoring=scorer,refit=True,n_jobs=-1)
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
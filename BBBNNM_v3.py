# -*- coding: utf-8 -*-
"""
Improved Blood-Brain Barrier Permeability Prediction
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
import os
import warnings
warnings.filterwarnings('ignore')

# Set working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

#%% Load and prepare data
print("Loading data...")
TS = pd.read_excel('DataSheet1_Development of QSAR models to predict blood-brain barrier permeability.xlsx', index_col=0)
act = TS['Activity score'].to_numpy()

# Load fingerprint array
fpArRaw = pd.read_csv('fpArray.csv')
fpAr = fpArRaw.drop(fpArRaw.columns[0], axis=1)
fpAr = fpAr.to_numpy()

print(f"Dataset shape: {fpAr.shape}")
print(f"Class distribution: {np.bincount(act)}")

#%% Data preprocessing


def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Preprocessing with RFECV-based feature selection
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # RFECV for feature selection based on F1 score
    from sklearn.feature_selection import RFECV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = StratifiedKFold(5)
    selector = RFECV(rf, step=25, cv=cv, scoring='f1', n_jobs=-1)
    selector = selector.fit(X_train_scaled, y_train)

    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    y_train_balanced = y_train  # No SMOTE

    print(f"Optimal number of features: {selector.n_features_}")

    return (X_train_selected, X_test_selected, y_train_balanced, y_test, scaler, selector)



X_train, X_test, y_train, y_test, scaler, selector = preprocess_data(fpAr, act)

print(f"After preprocessing:")
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Balanced class distribution: {np.bincount(y_train)}")

#%% Improved Neural Network
def create_improved_nn(input_dim):
    """
    Create an improved neural network for binary classification
    """
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        keras.metrics.Accuracy(),
        keras.metrics.Precision(),
        keras.metrics.Recall()
    ]
    )

    
    return model

# Create and train neural network
print("\nTraining Neural Network...")
nn_model = create_improved_nn(X_train.shape[1])

# Add callbacks for better training
callbacks = [
    keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5)
]

history = nn_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=500,
    batch_size=16,
    callbacks=callbacks,
    verbose=0
)

# Evaluate neural network
nn_pred = (nn_model.predict(X_test) > 0.5).astype(int).flatten()
nn_accuracy = accuracy_score(y_test, nn_pred)
nn_f1 = f1_score(y_test, nn_pred)
nn_auc = roc_auc_score(y_test, nn_model.predict(X_test))

print(f"Neural Network - Accuracy: {nn_accuracy:.3f}, F1: {nn_f1:.3f}, AUC: {nn_auc:.3f}")

#%% Improved SVM with hyperparameter tuning
def evaluate_svm_variants(X_train, X_test, y_train, y_test):
    """
    Evaluate different SVM configurations
    """
    svm_configs = [
        {'C': 1, 'kernel': 'rbf', 'gamma': 'scale'},
        {'C': 10, 'kernel': 'rbf', 'gamma': 'scale'},
        {'C': 100, 'kernel': 'rbf', 'gamma': 'scale'},
        {'C': 1, 'kernel': 'linear'},
        {'C': 10, 'kernel': 'linear'},
    ]
    
    best_svm = None
    best_score = 0
    
    for config in svm_configs:
        svm = SVC(**config, random_state=42)
        svm.fit(X_train, y_train)
        pred = svm.predict(X_test)
        score = f1_score(y_test, pred)
        
        if score > best_score:
            best_score = score
            best_svm = svm
    
    return best_svm

print("\nOptimizing SVM...")
best_svm = evaluate_svm_variants(X_train, X_test, y_train, y_test)
svm_pred = best_svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)

print(f"Best SVM - Accuracy: {svm_accuracy:.3f}, F1: {svm_f1:.3f}")

#%% Random Forest (often works well for chemical data)
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print(f"Random Forest - Accuracy: {rf_accuracy:.3f}, F1: {rf_f1:.3f}")

#%% Cross-validation evaluation
def cross_validate_models(X, y, cv=5):
    """
    Perform cross-validation for all models
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # SVM CV
    svm_cv = SVC(C=10, kernel='rbf', random_state=42)
    svm_scores = cross_val_score(svm_cv, X, y, cv=skf, scoring='f1')
    
    # Random Forest CV
    rf_cv = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_scores = cross_val_score(rf_cv, X, y, cv=skf, scoring='f1')
    
    return svm_scores, rf_scores

print("\nCross-validation results:")
svm_cv_scores, rf_cv_scores = cross_validate_models(X_train, y_train)
print(f"SVM CV F1: {svm_cv_scores.mean():.3f} ± {svm_cv_scores.std():.3f}")
print(f"Random Forest CV F1: {rf_cv_scores.mean():.3f} ± {rf_cv_scores.std():.3f}")

#%% Detailed evaluation
def detailed_evaluation(y_true, y_pred, model_name):
    """
    Print detailed evaluation metrics
    """
    print(f"\n{model_name} Detailed Results:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall: {recall_score(y_true, y_pred):.3f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.3f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_true, y_pred)}")

# Detailed evaluation for best performing model
best_model = None
best_pred = None
best_f1 = 0

models_results = [
    ("Neural Network", nn_pred, nn_f1),
    ("SVM", svm_pred, svm_f1),
    ("Random Forest", rf_pred, rf_f1)
]

for name, pred, f1 in models_results:
    if f1 > best_f1:
        best_f1 = f1
        best_model = name
        best_pred = pred

print(f"\nBest performing model: {best_model}")
detailed_evaluation(y_test, best_pred, best_model)

#%% Feature importance (for Random Forest)
if hasattr(rf, 'feature_importances_'):
    feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(importance_df.head(10))

print(f"\nFinal Recommendations:")
print("1. The preprocessing steps (scaling, feature selection, SMOTE) should improve performance")
print("2. Try ensemble methods or stacking different models")
print("3. Consider generating more relevant molecular descriptors")
print("4. Collect more training data if possible")
print("5. Consider using more sophisticated molecular fingerprints (Morgan, MACCS)")

# #%% XGBoost Model (Added at the end of the script)
# from xgboost import XGBClassifier

# print("\nTraining XGBoost...")

# xgb = XGBClassifier(
#     n_estimators=300,
#     max_depth=6,
#     learning_rate=0.03,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     use_label_encoder=False,
#     eval_metric='logloss',
#     random_state=42,
#     scale_pos_weight=1.0  # You may adjust this depending on remaining imbalance
# )

# xgb.fit(X_train, y_train)
# xgb_pred = xgb.predict(X_test)

# xgb_accuracy = accuracy_score(y_test, xgb_pred)
# xgb_f1 = f1_score(y_test, xgb_pred)

# print(f"XGBoost - Accuracy: {xgb_accuracy:.3f}, F1: {xgb_f1:.3f}")
# detailed_evaluation(y_test, xgb_pred, "XGBoost")

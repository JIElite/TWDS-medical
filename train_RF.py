import os
import joblib
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,
                             recall_score,
                             precision_score,
                             roc_auc_score)
import mlflow

from clean import clean_IDATE


# Parameters
MODEL = RandomForestClassifier.__name__
N_ESTIMATORS = 100
N_JOBS = 16
MAX_DEPTH = 20
MIN_SAMPLES_LEAF = 5
VALIDATION_METHOD = train_test_split.__name__
VAL_SIZE = 0.1
SHUFFLE_SEED = SPLIT_SEED = 42
DATA_SOURCE = './merged_data/brfss_combine_cleaned.csv'
TARGET = 'ADDEPEV3'


MLFLOW = True
exp_params = {
    'model': MODEL,
    'validation': VALIDATION_METHOD,
    'val_size': VAL_SIZE,
    'data_source': DATA_SOURCE,
    'shuffle_seed': SHUFFLE_SEED,
    'split_seed': SPLIT_SEED,
}
model_params = {
    'n_estimators': N_ESTIMATORS,
    'n_jobs': N_JOBS,
    'max_depth': MAX_DEPTH,
    'min_samples_leaf': MIN_SAMPLES_LEAF,
}

if MLFLOW:
    MLFLOW_SERVER_HOST = os.environ.get('MLFLOW_SERVER_HOST')
    EXPERIMENT_NAME = 'medical_project'
    RUN_NAME = 'Random Forest'
    mlflow.set_tracking_uri(MLFLOW_SERVER_HOST)
    mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
    mlflow.start_run(run_name=RUN_NAME)
    mlflow.log_params(exp_params)
    mlflow.log_params(model_params)
    # Log script
    SCRIPT_PATH = os.path.basename(__file__)
    mlflow.log_artifact(SCRIPT_PATH)

# Import data and Preprocessing
df = pd.read_csv(DATA_SOURCE)
df = clean_IDATE(df)

# Train and Val Split
target = TARGET
df = df.sample(frac=1, random_state=SHUFFLE_SEED)
X = df.drop(columns=[target])
y = df[target]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=VAL_SIZE, random_state=SPLIT_SEED)

# Modeling
clf = RandomForestClassifier(**model_params)
clf.fit(X_train, y_train)

# Evaluation
scoring_funcs = (accuracy_score, recall_score, precision_score, roc_auc_score)
scores = {}
for scoring_func in scoring_funcs:
    scoring_func_name = scoring_func.__name__
    scores[f'train_{scoring_func_name}'] = scoring_func(
        y_train, clf.predict(X_train))
    scores[f'val_{scoring_func_name}'] = scoring_func(
        y_val, clf.predict(X_val))
print(scores)

if MLFLOW:
    mlflow.log_metrics(scores)

# Save Model
MODEL_DIR = './models/'
MODEL_SUFFIX = str(datetime.now())
MODEL_PATH = f'{MODEL_DIR}model_{MODEL_SUFFIX}.pkl'
joblib.dump(clf, MODEL_PATH)
if MLFLOW:
    mlflow.log_artifact(MODEL_PATH)
    mlflow.end_run()

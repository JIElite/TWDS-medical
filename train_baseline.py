import os
import joblib
from datetime import datetime

import pandas as pd
from sklearn.model_selection import (
    cross_val_score, train_test_split)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score,
                             cohen_kappa_score,
                             confusion_matrix)
import mlflow

from clean import clean_IDATE


# Parameters
MODEL = DecisionTreeClassifier.__name__
MAX_DEPTH = 20
MIN_SAMPLES_LEAF = 20
VALIDATION_METHOD = train_test_split.__name__
VAL_SIZE = 0.1
SHUFFLE_SEED = SPLIT_SEED = 42
DATA_SOURCE = './merged_data/brfss_combine_cleaned.csv'
TARGET = 'ADDEPEV3'


MLFLOW = True
if MLFLOW:
    MLFLOW_SERVER_HOST = os.environ.get('MLFLOW_SERVER_HOST')
    EXPERIMENT_NAME = 'medical_project'
    RUN_NAME = 'DT Baseline'
    mlflow.set_tracking_uri(MLFLOW_SERVER_HOST)
    mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
    mlflow.start_run(run_name=RUN_NAME)
    mlflow.log_params({
        'model': MODEL,
        'max_depth': MAX_DEPTH,
        'min_samples_leaf': MIN_SAMPLES_LEAF,
        'validation': VALIDATION_METHOD,
        'val_size': VAL_SIZE,
        'data_source': DATA_SOURCE,
        'shuffle_seed': SHUFFLE_SEED,
        'split_seed': SPLIT_SEED,
    })
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
clf = DecisionTreeClassifier(
    max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF)
clf.fit(X_train, y_train)

# Evaluation
score_train = accuracy_score(y_train, clf.predict(X_train))
score_val = accuracy_score(y_val, clf.predict(X_val))
if MLFLOW:
    mlflow.log_metrics({
        f'train_{accuracy_score.__name__}': score_train,
        f'val_{accuracy_score.__name__}': score_val,
    })

# Save Model
MODEL_DIR = './models/'
MODEL_SUFFIX = str(datetime.now())
MODEL_PATH = f'{MODEL_DIR}model_{MODEL_SUFFIX}.pkl'
joblib.dump(clf, MODEL_PATH)
if MLFLOW:
    mlflow.log_artifact(MODEL_PATH)
    mlflow.end_run()

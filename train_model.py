from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

from model_training import train_model


MLFLOW = True
SAVE_MODEL = True
EVAL_TESTING = True

if __name__ == "__main__":
    exp_params = {
        "run_name": "Train Random Forest",
        "model_type": RandomForestClassifier.__name__,
        "training_data": "./merged_data/brfss_combine_train.csv",
        "testing_data": "./merged_data/brfss_combine_test.csv",
        "shuffle_seed": 42,
        "train_tests_split_seed": 42,
        "val_size": 0.1,
        "target": "ADDEPEV3",
        "prob_threshold": 0.3,
    }
    model_params = {
        "n_estimators": 100,
        "n_jobs": 16,
        "max_depth": 20,
        "min_samples_leaf": 10,
    }
    model_class = RandomForestClassifier
    scoring = (accuracy_score, recall_score, precision_score, roc_auc_score)

    # NOTICE: We should only evalute the testing set performance once
    # use eval_testing=False for tuning hyperparameters
    # use eval_testing=True for reporting final performance for a specfic model
    train_model(
        model_class,
        model_params,
        exp_params,
        scoring,
        use_mlflow=MLFLOW,
        save_model=SAVE_MODEL,
        eval_testing=EVAL_TESTING,
    )

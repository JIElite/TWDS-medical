from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

from model_training import train_model_cv
from eval import scoring_maps


MLFLOW = True
EVAL_TESTING = False


if __name__ == "__main__":
    exp_params = {
        "run_name": "Train Random Forest",
        "model_type": RandomForestClassifier.__name__,
        "training_data": "./merged_data/brfss_combine_train.csv",
        "testing_data": "./merged_data/brfss_combine_test.csv",
        "shuffle_seed": 42,
        "train_tests_split_seed": 42,
        "target": "ADDEPEV3",
        "prob_threshold": 0.3,
        "model_dir": "./models/",
    }
    model_params = {
        "n_estimators": 100,
        "n_jobs": 16,
        "max_depth": 20,
        "min_samples_leaf": 10,
    }
    scoring_funcs = (accuracy_score, recall_score, precision_score, roc_auc_score)
    scoring = [scoring_maps[metric_func] for metric_func in scoring_funcs]
    cv_params = {
        "n_jobs": 16,
        "cv": 5,
        "scoring": scoring,
        "return_train_score": True,
        "verbose": True,
    }
    model_class = RandomForestClassifier

    # NOTICE: We should only evalute the testing set performance once
    # use eval_testing=False for tuning hyperparameters
    # use eval_testing=True for reporting final performance for a specfic model
    train_model_cv(
        model_class,
        model_params,
        exp_params,
        cv_params,
        scoring_funcs,
        use_mlflow=MLFLOW,
        eval_testing=EVAL_TESTING,
    )

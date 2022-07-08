from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
    f1_score,
)

from flare.model_training import LightGBMTrainer
from flare.eval import Evaluator
from flare.data import LightGBMDataPreparer0708

MLFLOW = True
SAVE_MODEL = True
EVAL_TESTING = True


if __name__ == "__main__":
    data_preparer_class = LightGBMDataPreparer0708
    data_preparer = data_preparer_class()

    exp_params = {
        "run_name": "LGBM",
        "model_type": LGBMClassifier.__name__,
        "training_data": "./merged_data/brfss_combine_train_v2_important_20220708.csv",
        "testing_data": "./merged_data/brfss_combine_test_v2_important_20220708.csv",
        "shuffle_seed": 42,
        "train_tests_split_seed": 42,
        "val_size": 0.1,
        "target": "ADDEPEV3",
        "prob_threshold": 0.3,
        "model_dir": "./models/",
        "data_preparer": data_preparer_class.__name__,
    }
    model_params = {
        "n_estimators": 120,
        "n_jobs": -1,
        "max_depth": 20,
        "objective": "binary",
        "subsample": 0.8,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "reg_lambda": 0.0,
    }
    print(exp_params)
    print(model_params)
    model_class = LGBMClassifier
    scoring_funcs = (
        accuracy_score,
        recall_score,
        precision_score,
        roc_auc_score,
        f1_score,
    )
    evaluator = Evaluator(
        scoring_funcs=scoring_funcs,
        prob_threshold=exp_params["prob_threshold"],
        use_mlflow=MLFLOW,
    )
    # NOTICE: We should only evalute the testing set performance once
    # use eval_testing=False for tuning hyperparameters
    # use eval_testing=True for reporting final performance for a specfic model
    trainer = LightGBMTrainer(
        model_class,
        model_params,
        exp_params=exp_params,
        data_preparer=data_preparer,
        scoring_funcs=scoring_funcs,
        evaluator=evaluator,
        eval_testing=EVAL_TESTING,
        save_trained_model=SAVE_MODEL,
        save_testing_model=SAVE_MODEL,
        use_mlflow=MLFLOW,
    )
    trainer.run()

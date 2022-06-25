from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

from model_training import LightGBMTrainer, LightGBMCVTrainer
from eval import Evaluator, scoring_maps

MLFLOW = True
SAVE_MODEL = True
EVAL_TESTING = False


if __name__ == "__main__":
    exp_params = {
        "run_name": "LGBM",
        "model_type": LGBMClassifier.__name__,
        "training_data": "./merged_data/brfss_combine_train_v2.csv",
        "testing_data": "./merged_data/brfss_combine_test_v2.csv",
        "shuffle_seed": 42,
        "train_tests_split_seed": 42,
        "val_size": 0.1,
        "target": "ADDEPEV3",
        "prob_threshold": 0.3,
        "model_dir": "./models/",
    }
    model_params = {
        "n_estimators": 100,
        "n_jobs": 16,
        "max_depth": 20,
        "objective": "binary",
    }
    model_class = LGBMClassifier
    scoring_funcs = (accuracy_score, recall_score, precision_score, roc_auc_score)
    scoring = [scoring_maps[metric_func] for metric_func in scoring_funcs]
    evaluator = Evaluator(scoring_funcs=scoring_funcs, use_mlflow=MLFLOW)
    cv_params = {
        "n_jobs": 16,
        "cv": 5,
        "scoring": scoring,
        "return_train_score": True,
        "verbose": True,
    }
    # NOTICE: We should only evalute the testing set performance once
    # use eval_testing=False for tuning hyperparameters
    # use eval_testing=True for reporting final performance for a specfic model
    trainer = LightGBMTrainer(
        model_class,
        model_params,
        exp_params=exp_params,
        scoring_funcs=scoring_funcs,
        evaluator=evaluator,
        save_trained_model=SAVE_MODEL,
        use_mlflow=MLFLOW,
    )
    # trainer = LightGBMCVTrainer(
    #     model_class,
    #     model_params,
    #     exp_params=exp_params,
    #     cv_params=cv_params,
    #     scoring_funcs=scoring_funcs,
    #     evaluator=evaluator,
    #     eval_testing=EVAL_TESTING,
    #     use_mlflow=MLFLOW,
    #     save_testing_model=SAVE_MODEL,
    # )
    trainer.run()

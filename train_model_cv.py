from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

from flare.model_training import CVTrainer
from flare.eval import scoring_maps, Evaluator


MLFLOW = True
EVAL_TESTING = True
SAVE_TESTING_MODEL = True


if __name__ == "__main__":
    exp_params = {
        "run_name": "Train Random Forest",
        "model_type": RandomForestClassifier.__name__,
        "training_data": "./merged_data/brfss_combine_train_v2.csv",
        "testing_data": "./merged_data/brfss_combine_test_v2.csv",
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
    evaluator = Evaluator(
        scoring_funcs,
        prob_threshold=exp_params.get("prob_threshold", None),
        use_mlflow=MLFLOW,
    )
    cv_params = {
        "n_jobs": 16,
        "cv": 5,
        "scoring": scoring,
        "return_train_score": True,
        "verbose": True,
    }
    model_class = RandomForestClassifier

    trainer = CVTrainer(
        model_class=model_class,
        model_params=model_params,
        exp_params=exp_params,
        cv_params=cv_params,
        scoring_funcs=scoring_funcs,
        evaluator=evaluator,
        eval_testing=EVAL_TESTING,
        save_testing_model=SAVE_TESTING_MODEL,
        use_mlflow=MLFLOW,
    )
    trainer.run()

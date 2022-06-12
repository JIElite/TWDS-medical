import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import split_features_target, preprocess_IDATE
from mlflow_utils import environment_setup
from eval import evaluate_builtin_metric, eval_roc_auc_display


def train_model(
    model_class,
    model_params,
    exp_params,
    scoring_funcs,
    use_mlflow=False,
    eval_testing=False,
    save_model=True,
    save_model_path=None,
):
    """
    Args:
        scoring: callable or list of string, Defaults to mean_absolute_percentage_error.
            please refer to https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
            for predefined values.
    """
    # Preprocess Dataset
    df_train = pd.read_csv(exp_params["training_data"])
    df_train = df_train.sample(
        frac=1, random_state=exp_params.get("shuffle_seed", None)
    )
    df_train = preprocess_IDATE(df_train)

    target = exp_params["target"]
    val_size = exp_params["val_size"]
    X_train, y_train = split_features_target(df_train, target=target)
    y_train = y_train.replace(2, 0)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=exp_params.get("train_test_split_seed", None),
    )

    if use_mlflow:
        import mlflow

        environment_setup()
        mlflow.start_run(run_name=exp_params["run_name"])
        mlflow.log_params(exp_params)
        mlflow.log_params(model_params)
        # Log entry point script
        if "script" in exp_params:
            mlflow.log_artifact(exp_params["script"])

    model = model_class(**model_params)
    model.fit(X_train, y_train)
    prob_threshold = exp_params.get("prob_threshold", None)
    scores = {}
    scores.update(
        evaluate_builtin_metric(
            model,
            X_train,
            y_train,
            scoring_funcs,
            prob_threshold=prob_threshold,
            index_predix="train-",
        )
    )
    scores.update(
        evaluate_builtin_metric(
            model,
            X_val,
            y_val,
            scoring_funcs,
            prob_threshold=prob_threshold,
            index_predix="val-",
        )
    )

    eval_roc_auc_display(
        model, X_train, y_train, fig_path="train-auc.png", use_mlflow=use_mlflow
    )
    eval_roc_auc_display(
        model, X_val, y_val, fig_path="val-auc.png", use_mlflow=use_mlflow
    )

    # Evaluate on the testing dataset
    # TODO Refactoring
    if eval_testing:
        df_train = df_train.sample(
            frac=1, random_state=exp_params.get("shuffle_seed", None)
        )
        X_train, y_train = split_features_target(df_train, target=target)
        y_train = y_train.replace(2, 0)

        # Retrain model for testing the final performance
        model_test = model_class(**model_params)
        model_test.fit(X_train, y_train)

        df_test = pd.read_csv(exp_params["testing_data"])
        df_test = preprocess_IDATE(df_test)
        df_test = df_test.sample(
            frac=1, random_state=exp_params.get("shuffle_seed", None)
        )
        X_test, y_test = split_features_target(df_test, target=target)
        y_test = y_test.replace(2, 0)

        scores.update(
            evaluate_builtin_metric(
                model_test,
                X_test,
                y_test,
                scoring_funcs,
                prob_threshold=prob_threshold,
                index_predix="test-",
            )
        )
        eval_roc_auc_display(
            model_test, X_test, y_test, fig_path="test-auc.png", use_mlflow=use_mlflow
        )

    print(scores)

    # TODO implement after_metric_hook function
    # after_metric_hook.func(after_metric_hook.params, (X_train, y_train, X_val, y_val, X_test, y_test))

    if use_mlflow:
        mlflow.log_metrics(scores)

    # Save model
    if save_model:
        # TODO save model_test
        import joblib
        from datetime import datetime

        timestamp = str(datetime.now())

        MODEL_PATH = ""
        if save_model_path:
            MODEL_PATH = save_model_path
        else:
            MODEL_DIR = "./models/"
            MODEL_NAME = f'{exp_params["model_type"]}-{timestamp}.pkl'
            MODEL_PATH = f"{MODEL_DIR}{MODEL_NAME}"

        print("Saving model:", MODEL_PATH)
        joblib.dump(model, MODEL_PATH)
        if use_mlflow:
            mlflow.log_param("model_path", MODEL_PATH)
            mlflow.log_artifact(MODEL_PATH)

    if use_mlflow:
        mlflow.end_run()


def test():
    pass


def train_cv_model(
    model, model_params, exp_params, scoring, use_mlflow=False, save_model=True
):
    raise NotImplementedError

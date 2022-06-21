import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate

from preprocessing import split_features_target, preprocess_IDATE
from mlflow_utils import environment_setup
from eval import (
    evaluate_builtin_metric,
    eval_roc_auc_display,
    convert_cv_scores_to_logging_scores,
)
from utils import save_model


def train_model(
    model_class,
    model_params,
    exp_params,
    scoring_funcs,
    use_mlflow=False,
    eval_testing=False,
    save_trained_model=False,
    save_testing_model=False,
    model_path=None,
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
            mlflow.cv_params.scoring_funcslog_artifact(exp_params["script"])

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
    if eval_testing:
        testing_scores, model_testing = train_and_eval_testing_set(
            model_class, model_params, exp_params, scoring_funcs, use_mlflow
        )
        scores.update(testing_scores)

    # Show performance
    print(scores)
    if use_mlflow:
        mlflow.log_metrics(scores)

    # Save model
    if save_trained_model:
        save_model(
            model,
            mode="training",
            model_path=model_path,
            exp_params=exp_params,
            use_mlflow=use_mlflow,
            verbose=True,
        )
    if save_testing_model:
        save_model(
            model_testing,
            mode="testing",
            model_path=model_path,
            exp_params=exp_params,
            use_mlflow=use_mlflow,
            verbose=True,
        )

    if use_mlflow:
        mlflow.end_run()


def train_model_cv(
    model_class,
    model_params,
    exp_params,
    cv_params,
    scoring_funcs,
    use_mlflow=False,
    eval_testing=False,
    save_testing_model=False,
    save_model_path=None,
):
    df_train = pd.read_csv(exp_params["training_data"])
    df_train = df_train.sample(
        frac=1, random_state=exp_params.get("shuffle_seed", None)
    )
    df_train = preprocess_IDATE(df_train)

    target = exp_params["target"]
    X_train, y_train = split_features_target(df_train, target=target)
    y_train = y_train.replace(2, 0)

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
    scores = cross_validate(
        model,
        X_train,
        y_train,
        **cv_params,
    )
    scores = {
        k: v
        for k, v in scores.items()
        if str.startswith(k, "train") or str.startswith(k, "test")
    }
    # The sklearn cross_validate only support string-like scoring,
    # when evaluating multiple metrics, so we need to convert the
    # result from cross_validate to meet our logging format.
    scores = convert_cv_scores_to_logging_scores(scores)

    if eval_testing:
        testing_scores, model_testing = train_and_eval_testing_set(
            model_class, model_params, exp_params, scoring_funcs, use_mlflow
        )
        scores.update(testing_scores)

        if save_testing_model:
            save_model(
                model_testing,
                mode="testing",
                model_path=save_model_path,
                exp_params=exp_params,
                use_mlflow=use_mlflow,
                verbose=True,
            )

    # Show performance
    print(scores)
    if use_mlflow:
        mlflow.log_metrics(scores)

    if use_mlflow:
        mlflow.end_run()


def train_and_eval_testing_set(
    model_class, model_params, exp_params, scoring_funcs, use_mlflow
):
    target = exp_params["target"]
    prob_threshold = exp_params.get("prob_threshold", None)

    df_train = pd.read_csv(exp_params["training_data"])
    df_train = df_train.sample(
        frac=1, random_state=exp_params.get("shuffle_seed", None)
    )
    df_train = preprocess_IDATE(df_train)
    X_train, y_train = split_features_target(df_train, target=target)
    y_train = y_train.replace(2, 0)

    # Retrain model for testing the final performance
    model_test = model_class(**model_params)
    model_test.fit(X_train, y_train)

    df_test = pd.read_csv(exp_params["testing_data"])
    df_test = preprocess_IDATE(df_test)
    df_test = df_test.sample(frac=1, random_state=exp_params.get("shuffle_seed", None))
    X_test, y_test = split_features_target(df_test, target=target)
    y_test = y_test.replace(2, 0)

    scores = {}
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

    return scores, model_test

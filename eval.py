def eval_roc_auc_display(estimator, X, y, fig_path, use_mlflow=False):
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay

    RocCurveDisplay.from_estimator(estimator, X, y)
    plt.savefig(fig_path)

    if use_mlflow:
        import mlflow

        mlflow.log_artifact(fig_path)


def evaluate_builtin_metric(
    model, X, y, scoring_funcs, prob_threshold=None, index_predix=""
):
    scores = {}
    y_pred = None
    if prob_threshold:
        y_pred = model.predict_proba(X)[:, 1] > prob_threshold
    else:
        y_pred = model.predict(X)

    for scoring_func in scoring_funcs:
        scoring_func_name = scoring_func.__name__
        scores[index_predix + scoring_func_name] = scoring_func(y, y_pred)
    return scores

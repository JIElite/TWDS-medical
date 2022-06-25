import logging

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
    f1_score,
)
from sklearn.metrics import RocCurveDisplay
import mlflow


scoring_maps = {
    accuracy_score: "accuracy",
    recall_score: "recall",
    precision_score: "precision",
    roc_auc_score: "roc_auc",
    f1_score: "f1",
}

scoring_reverse_maps = {
    "accuracy": accuracy_score.__name__,
    "recall": recall_score.__name__,
    "precision": precision_score.__name__,
    "roc_auc": roc_auc_score.__name__,
    "f1": f1_score.__name__,
}


class Evaluator:
    def __init__(self, scoring_funcs, prob_threshold=None, use_mlflow=False):
        self.scoring_funcs = scoring_funcs
        self.prob_threshold = prob_threshold
        self.use_mlflow = use_mlflow

    def evaluate(self, model, X, y, prefix=""):
        scores = self._evaluate_builtin_metric(
            model,
            X,
            y,
            index_prefix=prefix,
        )
        if self.use_mlflow:
            mlflow.log_metrics(scores)

        return scores

    def eval_roc_auc_display(self, estimator, X, y, fig_path):
        RocCurveDisplay.from_estimator(estimator, X, y)
        plt.savefig(fig_path)
        plt.close()
        if self.use_mlflow:
            mlflow.log_artifact(fig_path)

    def _evaluate_builtin_metric(self, model, X, y, index_prefix=""):
        scores = {}
        y_pred = None

        if self.prob_threshold:
            y_pred = model.predict_proba(X)[:, 1] > self.prob_threshold
        else:
            y_pred = model.predict(X)

        for scoring_func in self.scoring_funcs:
            scoring_func_name = scoring_func.__name__
            if scoring_func == roc_auc_score:
                scores[index_prefix + scoring_func_name] = scoring_func(
                    y, model.predict_proba(X)[:, 1]
                )
            else:
                scores[index_prefix + scoring_func_name] = scoring_func(y, y_pred)
        return scores


def rebuild_cv_metric_name(cv_score_metric, eval_mode):
    prefix = "train-"
    if eval_mode == "test":
        prefix = "val-"

    starting_idx = len(eval_mode) + 1
    cv_metric_name = cv_score_metric[starting_idx:]
    try:
        metric_name = f"{prefix}{scoring_reverse_maps[cv_metric_name]}"
    except KeyError:
        logging.error('Unknown metric name: "{}"'.format(cv_metric_name))
    else:
        return metric_name


def convert_cv_scores_to_logging_scores(cv_scores):
    logging_scores = {}
    for cv_score_metric, cv_score in cv_scores.items():
        if str.startswith(cv_score_metric, "train"):
            eval_mode = "train"
        elif str.startswith(cv_score_metric, "test"):
            eval_mode = "test"
        else:
            raise ValueError(
                "Unknown cv_score_metric prefix: {}".format(cv_score_metric)
            )
        metric_name = rebuild_cv_metric_name(cv_score_metric, eval_mode)
        logging_scores[metric_name] = float(cv_score.mean())

    return logging_scores

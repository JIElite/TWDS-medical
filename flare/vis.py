import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
import mlflow
from sklearn.metrics import PrecisionRecallDisplay
import lightgbm as lgb


def plot_rf_importance(model, feature_names, n_features=25, use_mlflow=False):
    """Plotting the feature importances of Random Forest model

    Notice:
        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values).
        See sklearn.inspection.permutation_importance as an alternative.
        ref. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.feature_importances_

        TODO : use sklearn.inspection.permutation_importance
        TODO : use sns.barplot
    """
    feature_importances = model.feature_importances_.reshape(-1, 1)
    feature_names_array = feature_names.array.reshape(-1, 1)
    feature_importances_cat = np.concatenate(
        [feature_names_array, feature_importances], axis=1
    )
    feature_importances_cat.sort(axis=0)
    plt.barh(feature_importances_cat[-25:, 0], feature_importances_cat[-25:, 1])
    plt.savefig("feature-importances.png")
    plt.close()

    if use_mlflow:
        mlflow.log_artifact("feature-importances.png")


def plot_lgbm_importances(model, n_features=25, use_mlflow=False):
    ax = lgb.plot_importance(model, max_num_features=n_features)
    plt.savefig("importance.png")
    plt.close()
    if use_mlflow:
        mlflow.log_artifact("importance.png")


def plot_precision_recall_curve(model, X, y, filename=None, use_mlflow=False):
    display = PrecisionRecallDisplay.from_estimator(
        model, X, y, name="Precision-Recall-Curve"
    )
    if filename:
        plt.savefig(filename)
        plt.close()
        if use_mlflow:
            mlflow.log_artifact(filename)


def plot_shap_summary(model, X, filename, use_mlflow=False):
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)
    plt.savefig(filename)
    plt.close()
    if use_mlflow:
        mlflow.log_artifact(filename)

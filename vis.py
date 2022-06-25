import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mlflow


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

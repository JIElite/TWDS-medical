import os

import mlflow


DEFAULT_EXPRIMENT_NAME = "medical_project"


def environment_setup(experiment_name=DEFAULT_EXPRIMENT_NAME):
    SERVER_HOST = os.environ.get("MLFLOW_SERVER_HOST")
    mlflow.set_tracking_uri(SERVER_HOST)
    mlflow.set_experiment(experiment_name=experiment_name)

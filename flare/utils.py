from datetime import datetime
import joblib

import mlflow


def _simple_save_model(
    model, mode, save_model_path=None, exp_params=None, verbose=False
):
    timestamp = str(datetime.now())
    if save_model_path:
        MODEL_PATH = save_model_path
    else:
        MODEL_DIR = exp_params.get("model_dir", "./")
        MODEL_NAME = f'{exp_params["model_type"]}-{mode}-{timestamp}.pkl'
        MODEL_PATH = f"{MODEL_DIR}{MODEL_NAME}"

    if verbose:
        print("Saving model:", MODEL_PATH)

    joblib.dump(model, MODEL_PATH)
    return MODEL_PATH


def save_model(model, mode, model_path, exp_params, use_mlflow=False, verbose=False):
    model_path = _simple_save_model(
        model,
        mode=mode,
        save_model_path=model_path,
        exp_params=exp_params,
        verbose=verbose,
    )
    if use_mlflow:
        mlflow.log_param(f"model_path_{mode}", model_path)
        mlflow.log_artifact(model_path)

    return model_path

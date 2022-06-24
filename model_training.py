import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split, cross_validate

from preprocessing import split_features_target, preprocess_IDATE
from mlflow_utils import environment_setup
from eval import convert_cv_scores_to_logging_scores
from utils import save_model


class BaseTrainer:
    """The interface for training a model."""

    def __init__(self):
        pass

    def _prepare_data(self, mode):
        pass

    def run(self):
        pass


class Holdout_Trainer(BaseTrainer):
    def __init__(
        self,
        model_class,
        model_params,
        exp_params,
        scoring_funcs,
        evaluator,
        use_mlflow,
        eval_testing=False,
        save_trained_model=False,
        save_testing_model=False,
        model_path=None,
    ):
        self.model_class = model_class
        self.model_params = model_params
        self.exp_params = exp_params
        self.scoring_funcs = scoring_funcs
        self.evaluator = evaluator
        self.use_mlflow = use_mlflow
        self.eval_testing = eval_testing
        self.save_trained_model = save_trained_model
        self.save_testing_model = save_testing_model
        self.model_path = model_path

        if self.use_mlflow:
            environment_setup()
            mlflow.start_run(run_name=exp_params["run_name"])
            mlflow.log_params(exp_params)
            mlflow.log_params(model_params)
            # Log entry point script
            if "script" in exp_params:
                mlflow.cv_params.scoring_funcslog_artifact(exp_params["script"])

    def _prepare_data(self, mode="training_data"):
        df_train = pd.read_csv(self.exp_params[mode])
        df_train = df_train.sample(
            frac=1, random_state=self.exp_params.get("shuffle_seed", None)
        )
        df_train = preprocess_IDATE(df_train)
        X_train, y_train = split_features_target(
            df_train, target=self.exp_params["target"]
        )
        y_train = y_train.replace(2, 0)
        return X_train, y_train

    def run(self):
        X_train_original, y_train_original = self._prepare_data(mode="training_data")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_original,
            y_train_original,
            test_size=self.exp_params["val_size"],
            random_state=self.exp_params.get("train_test_split_seed", None),
        )
        model = self.train(X_train, y_train)
        if self.save_trained_model:
            save_model(
                model,
                mode="training",
                model_path=self.model_path,
                exp_params=self.exp_params,
                use_mlflow=self.use_mlflow,
                verbose=True,
            )
        train_scores = self.evaluator.evaluate(model, X_train, y_train, prefix="train-")
        self.evaluator.eval_roc_auc_display(
            model, X_train, y_train, fig_path="train-auc.png"
        )
        val_scores = self.evaluator.evaluate(model, X_val, y_val, prefix="val-")
        self.evaluator.eval_roc_auc_display(model, X_val, y_val, fig_path="val-auc.png")
        scores = {}
        scores.update(train_scores)
        scores.update(val_scores)

        if self.eval_testing:
            X_test, y_test = self._prepare_data(mode="testing_data")
            testing_scores, model_testing = self.eval_testing_data(
                X_train_original, y_train_original, X_test, y_test
            )
            scores.update(testing_scores)

            if self.save_testing_model:
                save_model(
                    model_testing,
                    mode="testing",
                    model_path=self.model_path,
                    exp_params=self.exp_params,
                    use_mlflow=self.use_mlflow,
                    verbose=True,
                )

        print(scores)
        if self.use_mlflow:
            mlflow.log_metrics(scores)
            mlflow.end_run()

    def train(self, X_train, y_train):
        model = self.model_class(**self.model_params)
        model.fit(X_train, y_train)
        return model

    def eval_testing_data(self, X_train, y_train, X_test, y_test):
        assert self.evaluator
        model_test = self.model_class(**self.model_params)
        model_test.fit(X_train, y_train)
        scores = self.evaluator.evaluate(model_test, X_test, y_test, prefix="test-")
        self.evaluator.eval_roc_auc_display(
            model_test, X_test, y_test, fig_path="test-auc.png"
        )
        return scores, model_test


class Trainer:
    def __init__(
        self,
        model_class,
        model_params,
        exp_params,
        scoring_funcs,
        evaluator=None,
        cv_params=None,
        eval_testing=False,
        save_testing_model=False,
        save_model_path=None,
        use_mlflow=False,
    ):
        self.model_class = model_class
        self.model_params = model_params
        self.scoring_funcs = scoring_funcs
        self.evaluator = evaluator
        self.exp_params = exp_params
        self.cv_params = cv_params
        self.eval_testing = eval_testing
        self.save_testing_model = save_testing_model
        self.save_model_path = save_model_path
        self.use_mlflow = use_mlflow

        if self.use_mlflow:

            environment_setup()
            mlflow.start_run(run_name=self.exp_params["run_name"])
            mlflow.log_params(self.exp_params)
            mlflow.log_params(self.model_params)
            # Log entry point script
            if "script" in self.exp_params:
                mlflow.log_artifact(exp_params["script"])

    def run(self):
        """Record the training flow."""
        X_train, y_train = self._prepare_data(data_mode="training_data")
        scores = self.train(X_train, y_train)

        if self.eval_testing:
            X_test, y_test = self._prepare_data(data_mode="testing_data")
            eval_testing_scores, model_testing = self.eval_testing_data(
                X_train, y_train, X_test, y_test
            )
            scores.update(eval_testing_scores)
            if self.save_testing_model:
                save_model(
                    model_testing,
                    mode="testing",
                    model_path=self.save_model_path,
                    exp_params=self.exp_params,
                    use_mlflow=self.use_mlflow,
                    verbose=True,
                )

        if self.use_mlflow:
            mlflow.end_run()
        print(scores)

    def _prepare_data(self, data_mode="training_data"):
        df_train = pd.read_csv(self.exp_params[data_mode])
        df_train = df_train.sample(
            frac=1, random_state=self.exp_params.get("shuffle_seed", None)
        )
        df_train = preprocess_IDATE(df_train)
        target = self.exp_params["target"]
        X, y = split_features_target(df_train, target=target)
        y = y.replace(2, 0)
        return X, y

    def train(self, X_train, y_train):
        self.model = self.model_class(**self.model_params)
        scores = cross_validate(self.model, X_train, y_train, **self.cv_params)
        scores = {
            k: v
            for k, v in scores.items()
            if str.startswith(k, "train") or str.startswith(k, "test")
        }

        # The sklearn cross_validate only support string-like scoring,
        # when evaluating multiple metrics, so we need to convert the
        # result from cross_validate to meet our logging format.True
        scores = convert_cv_scores_to_logging_scores(scores)
        return scores

    def eval_testing_data(self, X_train, y_train, X_test, y_test):
        assert self.evaluator
        model_test = self.model_class(**self.model_params)
        model_test.fit(X_train, y_train)
        scores = self.evaluator.evaluate(model_test, X_test, y_test, prefix="test-")
        self.evaluator.eval_roc_auc_display(
            model_test, X_test, y_test, fig_path="test-auc.png"
        )
        return scores, model_test

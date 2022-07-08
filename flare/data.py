import pandas as pd

from flare.preprocessing import preprocess_IDATE, split_features_target


class DataPreparerInterFace:
    def prepare_data(self, exp_params, data_mode="training_data"):
        pass


class BaseDataPreparer(DataPreparerInterFace):
    def prepare_data(self, exp_params, data_mode="training_data"):
        df_train = pd.read_csv(exp_params[data_mode])
        df_train = df_train.sample(
            frac=1, random_state=exp_params.get("shuffle_seed", None)
        )
        df_train = preprocess_IDATE(df_train)
        X_train, y_train = split_features_target(df_train, target=exp_params["target"])
        y_train = y_train.replace(2, 0)
        return X_train, y_train


class LightGBMDataPreparer(DataPreparerInterFace):
    def prepare_data(self, exp_params, data_mode="training_data"):
        df = pd.read_csv(exp_params[data_mode])
        df = df.sample(frac=1, random_state=exp_params.get("shuffle_seed", None))
        df = preprocess_IDATE(df)
        df.drop(columns=["Unnamed: 0"], inplace=True)

        X, y = split_features_target(df, target=exp_params["target"])
        y = y.replace(2, 0)
        return X, y


class LightGBMDataPreparer0708(DataPreparerInterFace):
    def prepare_data(self, exp_params, data_mode="training_data"):
        df = pd.read_csv(exp_params[data_mode])
        df = df.sample(frac=1, random_state=exp_params.get("shuffle_seed", None))
        df.drop(columns=["Unnamed: 0"], inplace=True)

        X, y = split_features_target(df, target=exp_params["target"])
        return X, y

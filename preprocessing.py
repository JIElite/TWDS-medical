DEPENDENT_TARGETS = "ADDEPEV3"


def split_features_target(df, target="ADDEPEV3", drop_columns=DEPENDENT_TARGETS):
    X = df.drop(columns=drop_columns)
    y = df[target]
    return X, y


def preprocess_IDATE(df):
    def convert_IDATE(x):
        """Convert Datetime 'yyyy-mm-dd' to integer yyyymm"""
        return x[:4] + x[5:7]

    df["IDATE"] = df["IDATE"].apply(convert_IDATE).astype(int)
    return df


class DataPreprocessor:
    def preprocess_df(self, df, mode=None):
        pass


class VaniilaLGBMPreprocessor(DataPreprocessor):
    def __init__(self, data_mode="training"):
        self.drop_columns = ["Unnamed: 0"]
        self.data_mode = data_mode

        if data_mode == "testing":
            self.drop_columns.append(DEPENDENT_TARGETS)

    def _prepare_data(self, df):
        """Preprocess the input data for LGBM"""
        df.drop(columns=self.drop_columns, inplace=True)
        df = preprocess_IDATE(df)
        return df

    def preprocess_df(self, df):
        return self._prepare_data(df)

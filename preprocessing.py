DEPENDENT_TARGETS = ('ADDEPEV3')


def split_features_target(df, target='ADDEPEV3', drop_columns=DEPENDENT_TARGETS):
    X = df.drop(columns=drop_columns)
    y = df[target]
    return X, y


def preprocess_IDATE(df):
    def convert_IDATE(x):
        """Convert Datetime 'yyyy-mm-dd' to integer yyyymm
        """
        return x[:4] + x[5:7]

    df['IDATE'] = df['IDATE'].apply(convert_IDATE).astype(int)
    return df

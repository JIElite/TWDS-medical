def clean_IDATE(df):
    def convert_IDATE(x):
        """Convert Datetime 'yyyy-mm-dd' to integer yyyymm
        """
        return x[:4] + x[5:7]

    df['IDATE'] = df['IDATE'].apply(convert_IDATE).astype(int)
    return df

import pandas as pd

from prepocess import (RESERVED_COLS,
                       preprocess_DIABAGE3,
                       preprocess_FEETCHK3,
                       preprocess_CHILDREN,
                       preprocess_QSTLANG,
                       preprocess_HHADULT,
                       preprocess_HTM4,
                       preprocess_MARIJAN1,
                       preprocess_HIVTSTD3_datetime,)


RAW_DATA_PATH = 'LLCP2020.csv'
PREPROCESSED_DATA_PATH = 'elichen.csv'
preprocess_funcs = {'DIABAGE3': preprocess_DIABAGE3,
                    'FEETCHK3': preprocess_FEETCHK3,
                    'CHILDREN': preprocess_CHILDREN,
                    'QSTLANG':  preprocess_QSTLANG,
                    'HHADULT':  preprocess_HHADULT,
                    'HTM4':     preprocess_HTM4,
                    'MARIJAN1': preprocess_MARIJAN1,
                    'HIVTSTD3': preprocess_HIVTSTD3_datetime}

df = pd.read_csv(RAW_DATA_PATH)
df_out = None
for preprocess_func in preprocess_funcs.values():
    df_out = preprocess_func(df)
df_out[RESERVED_COLS].to_csv(PREPROCESSED_DATA_PATH)

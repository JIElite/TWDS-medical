import pandas as pd

from prepocess import (preprocess_DIABAGE3,
                       preprocess_FEETCHK3,
                       preprocess_CHILDREN,
                       preprocess_QSTLANG,
                       preprocess_HHADULT,
                       preprocess_HTM4,
                       preprocess_MARIJAN1,
                       preprocess_HIVTSTD3_datetime,)


RAW_DATA_PATH = './data/LLCP2020.csv'
PREPROCESSED_DATA_PATH = './data/LLCP2020_elichen.csv'
preprocess_funcs = {'DIABAGE3': preprocess_DIABAGE3,
                    'FEETCHK3': preprocess_FEETCHK3,
                    'CHILDREN': preprocess_CHILDREN,
                    'QSTLANG':  preprocess_QSTLANG,
                    'HHADULT':  preprocess_HHADULT,
                    'HTM4':     preprocess_HTM4,
                    'MARIJAN1': preprocess_MARIJAN1,
                    'HIVTSTD3': preprocess_HIVTSTD3_datetime}

df = pd.read_csv(RAW_DATA_PATH)
for preprocess_func in preprocess_funcs.values():
    df = preprocess_func(df)
df.drop(columns=preprocess_funcs.keys())
df.to_csv(PREPROCESSED_DATA_PATH)
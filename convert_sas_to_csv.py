import pandas as pd

df = pd.read_sas('./data/LLCP2020.XPT')
df.to_csv('./data/LLCP2020.csv')
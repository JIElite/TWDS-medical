import pandas as pd

df = pd.read_sas('LLCP2020.XPT')
df.to_csv('LLCP2020.csv')

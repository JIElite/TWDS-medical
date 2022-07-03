import joblib

import pandas as pd

from preprocessing import VaniilaLGBMPreprocessor


data = pd.read_csv("merged_data/brfss_combine_test_v2.csv")
preprocessor = VaniilaLGBMPreprocessor(data_mode="testing")
data = preprocessor.preprocess_df(data)
model = joblib.load("models/LGBMClassifier-testing-2022-07-04 01:36:33.461375.pkl")
first_row = data.loc[[0]]
pred = model.predict(first_row)
print(pred)

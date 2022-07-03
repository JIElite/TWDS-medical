import joblib

import pandas as pd

from preprocessing import VaniilaLGBMPreprocessor


data = pd.read_csv("merged_data/brfss_combine_test_v2.csv")
preprocessor = VaniilaLGBMPreprocessor(data_mode="testing")
data = preprocessor.preprocess_df(data)
first_row = data.loc[[0]]
first_row = first_row.to_numpy()

# Numpy data is acceptable for inference
# we dont't need to provide the column information for the model
# In other words, we don't need to wrap the data in a DataFrame
model = joblib.load("models/LGBMClassifier-testing-2022-07-04 01:36:33.461375.pkl")
pred = model.predict(first_row)
print(pred)

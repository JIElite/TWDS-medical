import os

import joblib

import pandas as pd

from flare.preprocessing import VaniilaLGBMPreprocessor
from flare.inference import ProbabilisticBinaryClassifier


data = pd.read_csv("merged_data/brfss_combine_test_v2.csv")
preprocessor = VaniilaLGBMPreprocessor(data_mode="testing")
data = preprocessor.preprocess_df(data)

# Numpy data is acceptable for inference
# we dont't need to provide the column information for the model
# In other words, we don't need to wrap the data in a DataFrame
first_row = data.loc[[0]]
first_row = first_row.to_numpy()

# Here, we use an Inference Wrapper to adjust the threshold for inference
model = joblib.load("models/LGBMClassifier-testing-2022-07-04 01:36:33.461375.pkl")
model = ProbabilisticBinaryClassifier(model, prob_threshold=0.3)

# You can use sklearn-compatible API to do infernence
pred = model.predict(first_row)
print(pred)

# Or we also support Pytorch-compatible API (callable)
pred = model(first_row)
print(pred)

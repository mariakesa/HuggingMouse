import sys
import os
# Append the directory containing base.py to the system path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))
from pipelines import pipeline
from dotenv import load_dotenv
from single_trial_fs import MovieSingleTrialRegressionAnalysis
from transformers import ViTModel
from sklearn.linear_model import LinearRegression, Ridge

load_dotenv()

model = ViTModel.from_pretrained('google/vit-base-patch16-224')
regr_model = LinearRegression()
regr_model = Ridge(10)

pipe = pipeline("neural-activity-prediction",
                model=model,
                regression_model=regr_model,
                single_trial_f=MovieSingleTrialRegressionAnalysis(),
                test_set_size=0.25)
# pipe(511498742).dropna().plot()
# 511511001
pipe(511511001).dropna().plot()
# 646959386
# pipe(646959386).dropna().plot()

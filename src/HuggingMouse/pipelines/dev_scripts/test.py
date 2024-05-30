import sys
import os
# Append the directory containing base.py to the system path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))
from pipelines import pipeline
from dotenv import load_dotenv
from single_trial_fs import MovieSingleTrialRegressionAnalysis
from transformers import ViTModel

load_dotenv()

model = ViTModel.from_pretrained('google/vit-base-patch16-224')

pipe = pipeline("neuron-prediction", model=model,
                regression_model='house', single_trial_f=MovieSingleTrialRegressionAnalysis(), train_test_size=0.7)
pipe(511498742).filter_data('my_filter').plot('my_func')

print(pipe.eid_dict)

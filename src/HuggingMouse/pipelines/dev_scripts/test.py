import sys
import os
# Append the directory containing base.py to the system path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))
from pipelines import pipeline
from dotenv import load_dotenv

load_dotenv()

pipe = pipeline("neuron-prediction", model='car',
                regression_model='house', single_trial_f='tree')
pipe(511498742).filter_data('my_filter').plot('my_func')

print(pipe.eid_dict)

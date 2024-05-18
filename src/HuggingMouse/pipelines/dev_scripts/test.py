import sys
import os
# Append the directory containing base.py to the system path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))
from pipelines import pipeline

pipe = pipeline("neuron-prediction", model='car',
                regression_model='house', analysis_function='tree')
pipe(1234).filter_data('my_filter').plot('my_func')

# isort: on

pipe = pipeline("neuron-prediction", model='car',
                regression_model='house', analysis_function='tree')

pipe(1234).filter_data('my_filter').plot('my_func')

from dotenv import load_dotenv
import sys
sys.path.append("/home/maria/HuggingMouse/src/HuggingMouse/")
# from helper_functions import get_events, Test
from CEBRA import test
from HuggingMouse.pipelines.pipeline_tasks import pipeline
from dotenv import load_dotenv
from HuggingMouse.pipelines.single_trial_fs import MovieSingleTrialRegressionAnalysis
from transformers import ViTModel, CLIPVisionModel
from sklearn.linear_model import Ridge, Lasso

from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import r2_score
import numpy as np
import os
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from pathlib import Path
from HuggingMouse.utils import make_container_dict

load_dotenv()


def get_events(container_id):
    allen_cache_path = os.environ.get('HGMS_ALLEN_CACHE_PATH')
    print(allen_cache_path)
    boc = BrainObservatoryCache(manifest_file=str(
        Path(allen_cache_path) / Path('brain_observatory_manifest.json')))
    print(boc)
    eid = make_container_dict(boc)
    session_id = eid[container_id]['three_session_A']
    dataset = boc.get_ophys_experiment_data(ophys_experiment_id=session_id)
    # print(dataset.shape)
    dff_traces = dataset.get_dff_traces()[1]
    stimtable = dataset.get_stimulus_table('natural_movie_one')
    print(stimtable)


get_events(511511001)

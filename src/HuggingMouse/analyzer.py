from transformers import ViTImageProcessor, ViTModel
from regressors import VisionEmbeddingToNeuronsRegressor
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from allen_api_utilities import AllenExperimentUtility
#Starts here
import os
import pandas as pd
from get_config_params import get_cache_paths
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from utils import make_container_dict
import numpy as np


class MakeTrialAveragedData:
    def __init__(self):
        print('YES!')
        allen_cache_path, _ = get_cache_paths()
        self.boc = BrainObservatoryCache(manifest_file=str(Path(allen_cache_path) / Path('brain_observatory_manifest.json')))
        self.eid_dict = make_container_dict(self.boc)
        self.stimulus_session_dict= {
            'three_session_A': ['natural_movie_one', 'natural_movie_three'],
            'three_session_B': ['natural_movie_one'],
            'three_session_C': ['natural_movie_one', 'natural_movie_two'],
            'three_session_C2': ['natural_movie_one', 'natural_movie_two']
        }

    def make_single_session(self, container_id, session):
        session_eid  = self.eid_dict[container_id][session]
        dataset = self.boc.get_ophys_experiment_data(session_eid)
        dff_traces = dataset.get_dff_traces()[1]
        print('My SHAPE', dff_traces.shape)
        session_stimuli = self.stimulus_session_dict[session]
        session_dct = {}
        for s in session_stimuli:
            movie_stim_table = dataset.get_stimulus_table(s)
            stimuli = movie_stim_table.loc[movie_stim_table['repeat'] == trial]
            data=np.mean(dff_traces)
            #Code: session-->model-->stimulus-->trial
            var_exps = regression(data,self.regression_model)
            session_dct[str(sess)+'_'+str(s)+'_'+str(trial)] = var_exps
            #regression_vec_dct[str(sess)+'_'+str(m)+'_'+str(s)+'_'+str(trial)]=regr_vecs
        return session_dct#, regression_vec_dct
    
    def get_data(self,container_id):

        

if __name__=="__main__":
    # Check if the code is being executed in a documentation build environment
        # Skip API call in documentation build environment
    regression_model=Ridge(10)
    model = ViTModel.from_pretrained('google/vit-base-patch32-384')
    exps=AllenExperimentUtility()
    exps.view_all_imaged_areas()
    id=exps.experiment_container_ids_imaged_areas(['VISal'])[0]
    VisionEmbeddingToNeuronsRegressor(model,regression_model).execute(id)
    dim_reduction_model=PCA(n_components=3)
    trial_averaged_data=MakeTrialAveragedData(id).get_data()
    Visualizer(dim_reduction_model).visualize(trial_averaged_data)

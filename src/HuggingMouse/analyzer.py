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
from pathlib import Path


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

    def make_single_session_data(self, container_id, session):
        #Consider including the neuron id's so that the neurons are consistent across sessions
        session_eid  = self.eid_dict[container_id][session]
        dataset = self.boc.get_ophys_experiment_data(session_eid)
        dff_traces = dataset.get_dff_traces()[1]
        session_stimuli = self.stimulus_session_dict[session]
        session_dct = {}
        for s in session_stimuli:
            movie_stim_table = dataset.get_stimulus_table(s)
            trial_averaged_array=[]
            for frame in range(max(movie_stim_table['frame'])+1):
                stimuli=movie_stim_table.loc[movie_stim_table['frame']==frame]
                times=stimuli['start']
                vec=np.mean(dff_traces[:,times],axis=1)
                trial_averaged_array.append(vec)
            trial_averaged_array=np.array(trial_averaged_array).T
            session_dct[str(session)+'_'+str(s)]=trial_averaged_array
        return session_dct
    
    def get_data(self,container_id):
        for session in self.stimulus_session_dict.keys():
            #Some three sessions are only C and others only C2-- API twist. The try-except block takes care of 
            #these cases. 
            try:
                session_dct=self.make_single_session_data(container_id, session)
                print(session_dct)
            except:
                continue

        

if __name__=="__main__":
    # Check if the code is being executed in a documentation build environment
        # Skip API call in documentation build environment
    regression_model=Ridge(10)
    model = ViTModel.from_pretrained('google/vit-base-patch32-384')
    exps=AllenExperimentUtility()
    exps.view_all_imaged_areas()
    id=exps.experiment_container_ids_imaged_areas(['VISal'])[0]
    #VisionEmbeddingToNeuronsRegressor(model,regression_model).execute(id)
    #dim_reduction_model=PCA(n_components=3)
    trial_averaged_data=MakeTrialAveragedData().get_data(id)
    #Visualizer(dim_reduction_model).visualize(trial_averaged_data)

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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
            trial_averaged_array=np.array(trial_averaged_array)
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
        return session_dct

class Visualizer:
    def __init__(self,dim_reduction_model):
        self.dim_reduction_model=dim_reduction_model
        self.stimulus_session_dict= {
            'three_session_A': ['natural_movie_one', 'natural_movie_three'],
            'three_session_B': ['natural_movie_one'],
            'three_session_C': ['natural_movie_one', 'natural_movie_two'],
            'three_session_C2': ['natural_movie_one', 'natural_movie_two']
        }

    def info(self):
        print('These are all the possible session stimulus pairs: ', self.stimulus_session_dict)
    
    def visualize(self,trial_averaged_data, session, stimulus):
        try:
            X_new=self.dim_reduction_model.fit_transform(trial_averaged_data[str(session)+'_'+str(stimulus)])
            print('Hourray! X_new: ', X_new)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2], c=range(X_new.shape[0]), marker='o',cmap='bwr')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.title('3D Plot of X_new')
            plt.show()
        except:
            print('This session stimulus combination doesn\'t exist! in this experiment container')

        

if __name__=="__main__":
    # Check if the code is being executed in a documentation build environment
        # Skip API call in documentation build environment
    regression_model=Ridge(10)
    model = ViTModel.from_pretrained('google/vit-base-patch32-384')
    exps=AllenExperimentUtility()
    exps.view_all_imaged_areas()
    id=exps.experiment_container_ids_imaged_areas(['VISal'])[0]
    #VisionEmbeddingToNeuronsRegressor(model,regression_model).execute(id)
    dim_reduction_model=PCA(n_components=3)
    trial_averaged_data=MakeTrialAveragedData().get_data(id)
    visualizer=Visualizer(dim_reduction_model)
    visualizer.info()
    visualizer.visualize(trial_averaged_data,'three_session_A','natural_movie_one')

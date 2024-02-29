from transformers import ViTImageProcessor, ViTModel
from pathlib import Path
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import numpy as np
from get_config_params import get_cache_paths
import torch
import pickle
import os
from exceptions import CachePathNotSpecifiedError
import pandas as pd

class MakeEmbeddings:
    allen_cache_path, transformer_embedding_cache_path = get_cache_paths()
    #Experiments where these three types of movies were played
    session_A = 501704220  # This is three session A
    session_B = 501559087
    session_C = 501474098
    boc = BrainObservatoryCache(manifest_file=str(Path(allen_cache_path) / Path('brain_observatory_manifest.json')))
    raw_data_dct = {}
    movie_one_dataset = boc.get_ophys_experiment_data(session_A)
    raw_data_dct['movie_one'] = movie_one_dataset.get_stimulus_template('natural_movie_one')
    movie_two_dataset = boc.get_ophys_experiment_data(session_C)
    raw_data_dct['movie_two'] = movie_two_dataset.get_stimulus_template('natural_movie_two')
    movie_three_dataset = boc.get_ophys_experiment_data(session_A)
    raw_data_dct['movie_three'] = movie_three_dataset.get_stimulus_template('natural_movie_three')

    def __init__(self, processor, model, cache_data=True):
        self.processor = processor
        self.model = model
        self.cache_data=cache_data


    def process_stims(self, stims):
        n_stims = len(stims)
        #n_stims=10
        stims_dim = np.repeat(stims[:, np.newaxis, :, :], 3, axis=1)
        embeddings = np.empty((n_stims, 768))
        for i in range(n_stims):
            print(i)
            inputs = self.processor(images=stims_dim[i], return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls = outputs.pooler_output.squeeze().detach().numpy()
            embeddings[i, :] = cls
        return embeddings
    
    def save_to_cache(self, embeddings_dct):
        # Replace / with _ for valid file name
        model_string=self.model.name_or_path.replace('/', '_') 
        file_name = model_string+'_embeddings.pkl'
        # Pickle the dictionary
        save_path=Path(self.transformer_embedding_cache_path) / Path(file_name)
        with open(save_path, 'wb') as f:
            pickle.dump(embeddings_dct, f)

    def execute(self):
        embeddings_dct = {}
        for key in self.raw_data_dct.keys():
            print(self.raw_data_dct[key].shape)
            embeddings_dct[key] = self.process_stims(self.raw_data_dct[key])
        if self.cache_data:
            if not self.project_cache_path:
                raise CachePathNotSpecifiedError("No transforme embedding cache path specified in config.json!")
            elif not os.path.exists(self.project_cache_path):
                raise FileNotFoundError(f"Project cache path '{self.project_cache_path}' does not exist!")
            else:
                self.save_to_cache(embeddings_dct)
        return embeddings_dct

class AllenExperimentUtility:
    def __init__(self):
        allen_cache_path, transformer_embedding_cache_path = get_cache_paths()
        self.boc = BrainObservatoryCache(manifest_file=str(Path(allen_cache_path) / 'brain_observatory_manifest.json'))
        
    def view_all_imaged_areas(self):
        print(self.boc.get_all_targeted_structures())

    def view_all_cre_lines(self):
        print(self.boc.get_all_cre_lines())

    def imaged_area_info(self, imaged_area):
        print(0)

    def experiment_container_ids_imaged_areas(self, imaged_areas):
        experiment_containers=self.boc.get_experiment_containers(targeted_structures=imaged_areas)
        ecids=[exp_c['id'] for exp_c in experiment_containers]
        print('These are experimental containers\'s that contain query imaged areas: ',
              experiment_containers)
        print('These are experimental container id\'s corresponding to imaged areas', ecids)
        return ecids
    
from utils import make_container_dict, generate_random_state, regression, process_single_trial

class VisionEmbeddingToNeuronsRegressor:
    def __init__(self, model, regression_model):
        allen_cache_path, transformer_embedding_cache_path = get_cache_paths()
        self.model = model
        self.model_name_str = model.name_or_path
        self.processor = ViTImageProcessor.from_pretrained(self.model_name_str)
        self.model_prefix=self.model.name_or_path.replace('/', '_') 
        self.regression_model = regression_model
        self.regression_model_class = regression_model.__class__.__name__
        embedding_file_path = os.path.join(transformer_embedding_cache_path, f"{self.model_prefix}_embeddings.pkl")
        if not os.path.exists(embedding_file_path):
            self.embeddings=MakeEmbeddings(self.processor, self.model).execute()
        else:
            with open(embedding_file_path, 'rb') as f:
                self.embeddings = pickle.load(f)
        self.boc = BrainObservatoryCache(manifest_file=str(Path(allen_cache_path) / Path('brain_observatory_manifest.json')))
        self.eid_dict = make_container_dict(self.boc)
        self.stimulus_session_dict= {
            'three_session_A': ['natural_movie_one', 'natural_movie_three'],
            'three_session_B': ['natural_movie_one'],
            'three_session_C': ['natural_movie_one', 'natural_movie_two'],
            'three_session_C2': ['natural_movie_one', 'natural_movie_two']
        }
        print(self.embeddings.keys())
        self.random_state_dct=generate_random_state(seed=7, stimulus_session_dict=self.stimulus_session_dict)

    def make_regression_data(self, container_id, session):
        session_eid  = self.eid_dict[container_id][session]
        dataset = self.boc.get_ophys_experiment_data(session_eid)
        cell_ids = dataset.get_cell_specimen_ids()
        dff_traces = dataset.get_dff_traces()[1]
        session_stimuli = self.stimulus_session_dict[session]
        session_dct = pd.DataFrame()
        regression_vec_dct={}
        session_dct['cell_ids'] = cell_ids
        #regression_vec_dct['cell_ids'] = cell_ids
        #Compile the sessions into the same column to avoind NAN's
        #and make the data processing a bit easier
        if session=='three_session_C2':
            sess='three_session_C'
        else:
            sess=session
        for s in session_stimuli:
            movie_stim_table = dataset.get_stimulus_table(s)
            #Hack-- TODO
            embedding=self.embeddings[s[8:]]
            #There are only 10 trials in each session-stimulus pair
            for trial in range(10):
                random_state=self.random_state_dct[session][s][trial]
                data=process_single_trial(movie_stim_table, dff_traces, trial, embedding, random_state=random_state)
                #Code: session-->model-->stimulus-->trial
                var_exps = regression(data,self.regression_model)
                session_dct[str(sess)+'_'+str(s)+'_'+str(trial)] = var_exps
                #regression_vec_dct[str(sess)+'_'+str(m)+'_'+str(s)+'_'+str(trial)]=regr_vecs
        return session_dct#, regression_vec_dct
    
    def execute(self, container_id):
        for session in self.stimulus_session_dict.keys():
            #Some three sessions are only C and others only C2-- API twist. The try-except block takes care of 
            #these cases. 
            try:
                session_dct=self.make_regression_data(container_id, session)
                print(session_dct)
            except:
                continue
        




if __name__=="__main__":
    from sklearn.linear_model import LinearRegression, Ridge
    from transformers import ViTImageProcessor, ViTModel
    regression_model=Ridge(10)
    model = ViTModel.from_pretrained('google/vit-base-patch32-384')
    exps=AllenExperimentUtility()
    exps.view_all_imaged_areas()
    id=exps.experiment_container_ids_imaged_areas(['VISal'])[0]
    VisionEmbeddingToNeuronsRegressor(model,regression_model).execute(id)
    '''
    #Initialize model and processor
    #processor = ViTImageProcessor.from_pretrained('google/vit-base-patch32-384')
    #model = ViTModel.from_pretrained('google/vit-base-patch32-384')
    #model_name_str = model.name_or_path
    #print(model_name_str)
    #MakeEmbeddings(processor, model).execute()
    #boc = BrainObservatoryCache(manifest_file=str(Path("/media/maria/DATA/AllenData") / 'brain_observatory_manifest.json'))
    #experiment_container = boc.get_experiment_containers()
    #print(experiment_container)
    exps=AllenExperimentUtility()
    exps.view_all_imaged_areas()
    id=exps.experiment_container_ids_imaged_areas(['VISal'])[0]
    #API:
    from sklearn.linear_model import LinearRegression
    from transformers import ViTImageProcessor, ViTModel
    regression_model=LinearRegression()
    model = ViTModel.from_pretrained('google/vit-base-patch32-384')
    exps=AllenExperimentUtility()
    exps.view_all_imaged_areas()
    exp_id=exps.experiment_container_ids_imaged_areas(['VISal'])[0]
    VisionEmbeddingToNeuronsRegressor(model,regression_model).execute(exp_id, test_train_split_config)
    #RegressionMachine(model, regression_model).execute(exp_id, test_train_split_dct)
'''

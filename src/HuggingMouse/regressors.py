from transformers import AutoImageProcessor
from utils import make_container_dict, generate_random_state, regression, process_single_trial
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from get_config_params import get_cache_paths
from pathlib import Path
import pickle
import os
from make_embeddings import MakeEmbeddings
import pandas as pd
from sklearn.base import clone

class VisionEmbeddingToNeuronsRegressor:
    def __init__(self, model, regression_model):
        allen_cache_path, transformer_embedding_cache_path = get_cache_paths()
        self.model = model
        self.model_name_str = model.name_or_path
        self.processor = AutoImageProcessor.from_pretrained(self.model_name_str)
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
            embedding=self.embeddings[s]
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
            except Exception as e: 
                print(e)

from base import Pipeline
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from pathlib import Path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))
from HuggingMouse.utils import make_container_dict
import pandas as pd
from HuggingMouse.make_embeddings import MakeEmbeddings
import pickle


class NeuronPredictionPipeline(Pipeline):
    def __init__(self, **kwargs):
        self.model = kwargs['model']
        self.model_name_str = self.model.name_or_path
        self.model_prefix = self.model.name_or_path.replace('/', '_')
        self.regression_model = kwargs['regression_model']
        self.single_trial_f = kwargs['single_trial_f']
        self.test_set_size = kwargs.get('test_set_size', 0.7)
        allen_cache_path = os.environ.get('HGMS_ALLEN_CACHE_PATH')
        if allen_cache_path is None:
            pass
            # raise AllenCachePathNotSpecifiedError()
        transformer_embedding_cache_path = os.environ.get(
            'HGMS_TRANSF_EMBEDDING_PATH')
        if transformer_embedding_cache_path is None:
            pass
            # raise TransformerEmbeddingCachePathNotSpecifiedError()
        self.regr_analysis_results_cache = os.environ.get(
            'HGMS_REGR_ANAL_PATH')
        if self.regr_analysis_results_cache is None:
            pass
            # raise RegressionOutputCachePathNotSpecifiedError()
        self.boc = BrainObservatoryCache(manifest_file=str(
            Path(allen_cache_path) / Path('brain_observatory_manifest.json')))
        self.eid_dict = make_container_dict(self.boc)
        self.stimulus_session_dict = {
            'three_session_A': ['natural_movie_one', 'natural_movie_three'],
            'three_session_B': ['natural_movie_one'],
            'three_session_C': ['natural_movie_one', 'natural_movie_two'],
            'three_session_C2': ['natural_movie_one', 'natural_movie_two']
        }
        embedding_file_path = os.path.join(
            transformer_embedding_cache_path, f"{self.model_prefix}_embeddings.pkl")
        if not os.path.exists(embedding_file_path):
            self.embeddings = MakeEmbeddings(
                self.processor, self.model).execute()
        else:
            with open(embedding_file_path, 'rb') as f:
                self.embeddings = pickle.load(f)

    def __call__(self, container_id) -> str:
        output = 'boom'
        self.current_container = container_id
        for session, _ in self.stimulus_session_dict.items():
            try:
                session_dct = self.handle_single_session(container_id, session)
            except:
                print('Error')
        # self.single_trial_f(self.model, self.regression_model,
        # container_id)
        print(output)
        return self

    def handle_single_session(self, container_id, session):
        session_eid = self.eid_dict[container_id][session]
        dataset = self.boc.get_ophys_experiment_data(session_eid)
        cell_ids = dataset.get_cell_specimen_ids()
        dff_traces = dataset.get_dff_traces()[1]
        session_stimuli = self.stimulus_session_dict[session]
        session_dct = pd.DataFrame()
        session_dct['cell_ids'] = cell_ids
        # Compile the sessions into the same column to avoind NAN's
        # and make the data processing a bit easier
        if session == 'three_session_C2':
            sess = 'three_session_C'
        else:
            sess = session
        for s in session_stimuli:
            movie_stim_table = dataset.get_stimulus_table(s)
            embedding = self.embeddings[s]
            # There are only 10 trials in each session-stimulus pair
            data_dct = {
                'movie_stim_table': movie_stim_table,
                'dff_traces': dff_traces,
                'embedding': embedding,
                'test_set_size': self.test_set_size
            }
            for trial in range(10):
                data_dct['trial'] = trial
                return_dict = self.single_trial_f(**data_dct)

                # random_state = self.random_state_dct[session][s][trial]
                # data = process_single_trial(
                # movie_stim_table, dff_traces, trial, embedding, random_state=random_state)
                # Code: session-->model-->stimulus-->trial
                # scores = regression(data, self.regression_model, self.metrics)
                # for k in scores:
                # session_dct[str(sess) + '_' + str(s) + '_' +
                # str(trial) + '_' + k] = scores[k]
                # regression_vec_dct[str(sess)+'_'+str(m)+'_'+str(s)+'_'+str(trial)]=regr_vecs
        session_dct = {}
        return session_dct

    def plot(self, args=None):
        print('plotting')
        return self

    def filter_data(self, args=None):
        print('filtering')
        return self

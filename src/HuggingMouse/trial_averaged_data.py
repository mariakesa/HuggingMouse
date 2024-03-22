import os
import pandas as pd
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from HuggingMouse.utils import make_container_dict
import numpy as np
from pathlib import Path
from HuggingMouse.custom_exceptions import AllenCachePathNotSpecifiedError


class MakeTrialAveragedData:
    '''
    Average trial data from an experimental container ID for visualization using the
    VisualizerDimReduction class.
    '''

    def __init__(self):
        allen_cache_path = os.environ.get('HGMS_ALLEN_CACHE_PATH')
        if allen_cache_path is None:
            raise AllenCachePathNotSpecifiedError()
        self.boc = BrainObservatoryCache(manifest_file=str(
            Path(allen_cache_path) / Path('brain_observatory_manifest.json')))
        self.eid_dict = make_container_dict(self.boc)
        self.stimulus_session_dict = {
            'three_session_A': ['natural_movie_one', 'natural_movie_three'],
            'three_session_B': ['natural_movie_one'],
            'three_session_C': ['natural_movie_one', 'natural_movie_two'],
            'three_session_C2': ['natural_movie_one', 'natural_movie_two']
        }

    def make_single_session_data(self, container_id, session):
        # Consider including the neuron id's so that the neurons are consistent across sessions
        session_eid = self.eid_dict[container_id][session]
        dataset = self.boc.get_ophys_experiment_data(session_eid)
        dff_traces = dataset.get_dff_traces()[1]
        session_stimuli = self.stimulus_session_dict[session]
        session_dct = {}
        for s in session_stimuli:
            movie_stim_table = dataset.get_stimulus_table(s)
            trial_averaged_array = []
            for frame in range(max(movie_stim_table['frame'])+1):
                stimuli = movie_stim_table.loc[movie_stim_table['frame'] == frame]
                times = stimuli['start']
                vec = np.mean(dff_traces[:, times], axis=1)
                trial_averaged_array.append(vec)
            trial_averaged_array = np.array(trial_averaged_array)
            session_dct[str(s)] = trial_averaged_array
        return session_dct

    def get_data(self, container_id):
        all_sessions_dct = {}
        for session in self.stimulus_session_dict.keys():
            # Some three sessions are only C and others only C2-- API twist. The try-except block takes care of
            # these cases.
            try:
                session_dct = self.make_single_session_data(
                    container_id, session)
                all_sessions_dct[session] = session_dct
            except:
                continue
        return all_sessions_dct

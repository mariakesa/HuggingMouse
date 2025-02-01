from smolagents import tool
import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import numpy as np


@tool
def allen_engine(session: int) -> dict:

    """
    Allows you to construct a dictionary with neurons as keys
    for a session and the number of spikes in each 
    time interval for that neuron as values. The data comes
    from Allen Brain Observatory API.

    Args:
        session: int, Allen Brain Observatory session ID
    """    
    
    def get_spikes_in_intervals_optimized(spike_times, start_times, stop_times):
        spikes_in_intervals = {}

        # Convert start_times and stop_times to numpy arrays for faster operations
        start_times = np.array(start_times)
        stop_times = np.array(stop_times)

        # Loop through each neuron, but optimize the spike finding with vectorized operations
        for neuron_id, times in spike_times.items():
            # Convert times to a numpy array if it's not already
            times = np.array(times)
            
            # Use numpy's searchsorted to find the indices where the start and stop times would fit
            start_indices = np.searchsorted(times, start_times, side='left')
            stop_indices = np.searchsorted(times, stop_times, side='right')
            
            # Get the number of spikes in each interval by subtracting indices
            spikes_in_intervals[neuron_id] = stop_indices - start_indices
        
        return spikes_in_intervals
    output_dir = '/home/maria/AllenData'
    manifest_path = os.path.join(output_dir, "manifest.json")

    #functional_conn_session=[831882777]

    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    func_session = cache.get_session_data(session)

    stimuli_of_interest= ['natural_movie_one_more_repeats']

    stimuli_df=func_session.stimulus_presentations

    df_one_more_repeats = stimuli_df[stimuli_df['stimulus_name'] == 'natural_movie_one_more_repeats']

    start_times = df_one_more_repeats['start_time'].values
    stop_times = df_one_more_repeats['stop_time'].values

    spike_times=func_session.spike_times

    binned_spikes = get_spikes_in_intervals_optimized(spike_times, start_times, stop_times)

    return binned_spikes

    

import numpy as np
import pandas as pd

def make_container_dict(boc):
    '''
    Parses which experimental id's (values)
    correspond to which experiment containers (keys).
    '''
    experiment_container = boc.get_experiment_containers()
    container_ids=[dct['id'] for dct in experiment_container]
    eids=boc.get_ophys_experiments(experiment_container_ids=container_ids)
    df=pd.DataFrame(eids)
    reduced_df=df[['id', 'experiment_container_id', 'session_type']]
    grouped_df = reduced_df.groupby(['experiment_container_id', 'session_type'])['id'].agg(list).reset_index()
    eid_dict = {}
    for row in grouped_df.itertuples(index=False):
        container_id, session_type, ids = row
        if container_id not in eid_dict:
            eid_dict[container_id] = {}
        eid_dict[container_id][session_type] = ids[0]
    return eid_dict

def generate_random_state(seed, stimulus_session_dict):
    np.random.seed(seed)

    # Function to generate a random integer
    def generate_random_integer():
        return np.random.randint(1, 101)  # Generates a random integer between 1 and 100 (inclusive)

    # Create the main dictionary
    random_state_dct = {}

    # Populate the dictionary using stimulus_session_dict
    for session, stimuli_list in stimulus_session_dict.items():
        session_dict = {}
        for stimulus in stimuli_list:
            nested_dict = {trial: generate_random_integer() for trial in range(10)}
            session_dict[stimulus] = nested_dict
        random_state_dct[session] = session_dict
    return random_state_dct
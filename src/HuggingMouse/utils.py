
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
import hashlib


def process_single_trial(movie_stim_table, dff_traces, trial, embedding, random_state):
    stimuli = movie_stim_table.loc[movie_stim_table['repeat'] == trial]
    X_train, X_test, y_train_inds, y_test_inds = train_test_split(
        embedding, stimuli['start'].values, test_size=0.7, random_state=random_state)
    y_train = dff_traces[:, y_train_inds]
    y_test = dff_traces[:, y_test_inds]
    return {'y_train': y_train, 'y_test': y_test, 'X_train': X_train, 'X_test': X_test}


def regression(dat_dct, model, metrics):

    y_train, y_test, X_train, X_test = dat_dct['y_train'], dat_dct[
        'y_test'], dat_dct['X_train'], dat_dct['X_test']

    regr = clone(model)
    # Fit the model with scaled training features and target variable
    regr.fit(X_train, y_train.T)

    # Make predictions on scaled test features
    predictions = regr.predict(X_test)

    scores = {}
    for metric in metrics:
        neurons = []
        for i in range(0, y_test.shape[0]):
            neurons.append(metric(y_test.T[:, i], predictions[:, i]))
        scores[metric.__name__] = neurons
    return scores  # , regr.coef_.tolist()


def make_container_dict(boc):
    '''
    Parses which experimental id's (values)
    correspond to which experiment containers (keys).
    '''
    experiment_container = boc.get_experiment_containers()
    container_ids = [dct['id'] for dct in experiment_container]
    eids = boc.get_ophys_experiments(experiment_container_ids=container_ids)
    df = pd.DataFrame(eids)
    reduced_df = df[['id', 'experiment_container_id', 'session_type']]
    grouped_df = reduced_df.groupby(['experiment_container_id', 'session_type'])[
        'id'].agg(list).reset_index()
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
        # Generates a random integer between 1 and 100 (inclusive)
        return np.random.randint(1, 101)

    # Create the main dictionary
    random_state_dct = {}

    # Populate the dictionary using stimulus_session_dict
    for session, stimuli_list in stimulus_session_dict.items():
        session_dict = {}
        for stimulus in stimuli_list:
            nested_dict = {trial: generate_random_integer()
                           for trial in range(10)}
            session_dict[stimulus] = nested_dict
        random_state_dct[session] = session_dict
    return random_state_dct


def hash_df(df):
    # Convert DataFrame to a string representation
    df_str = df.to_string(index=False).encode('utf-8')

    # Calculate the SHA-1 hash
    sha1_hash = hashlib.sha1(df_str).hexdigest()

    return sha1_hash

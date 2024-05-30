from typing import Any
from sklearn.model_selection import train_test_split


class MovieSingleTrialRegressionAnalysis:
    def __init__(self):
        pass

    # , random_state):
    def train_test_split(self, movie_stim_table, dff_traces, trial, embedding, test_set_size):
        stimuli = movie_stim_table.loc[movie_stim_table['repeat'] == trial]
        X_train, X_test, y_train_inds, y_test_inds = train_test_split(
            embedding, stimuli['start'].values, test_size=test_set_size, random_state=7)  # , random_state=random_state)
        y_train = dff_traces[:, y_train_inds]
        y_test = dff_traces[:, y_test_inds]
        return {'y_train': y_train, 'y_test': y_test, 'X_train': X_train, 'X_test': X_test}

    def __call__(self, **kwargs: Any) -> Any:
        train_test_dict = self.train_test_split(
            kwargs['movie_stim_table'], kwargs['dff_traces'], kwargs['trial'], kwargs['embedding'], kwargs['test_set_size'])  # , random_state)

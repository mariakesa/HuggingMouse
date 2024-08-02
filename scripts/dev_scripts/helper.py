from dotenv import load_dotenv
import sys
sys.path.append("/home/maria/HuggingMouse/src/HuggingMouse/")
from helper_functions import get_events, Test
from HuggingMouse.pipelines.pipeline_tasks import pipeline
from dotenv import load_dotenv
from HuggingMouse.pipelines.single_trial_fs import MovieSingleTrialRegressionAnalysis
from transformers import ViTModel, CLIPVisionModel
from sklearn.linear_model import Ridge

from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import r2_score
import numpy as np


class MovieSingleTrialRegressionAnalysis_:
    def __init__(self):
        pass

    def train_test_split_interleaved(self, movie_stim_table, dff_traces, trial, embedding, test_set_size):
        '''
        From https://github.com/MouseLand/rastermap/blob/main/notebooks/tutorial.ipynb
        '''
        stimuli = movie_stim_table.loc[movie_stim_table['repeat'] == trial]
        n_time = stimuli.shape[0]
        n_segs = 20
        n_len = n_time / n_segs
        sinds = np.linspace(0, n_time - n_len, n_segs).astype(int)
        itest = (sinds[:, np.newaxis] +
                 np.arange(0, n_len * test_set_size, 1, int)).flatten()
        itrain = np.ones(n_time, "bool")
        itrain[itest] = 0
        itest = ~itrain
        train_inds = stimuli['start'].values[itrain]
        test_inds = stimuli['start'].values[itest]
        y_train = dff_traces[:, train_inds]
        y_test = dff_traces[:, test_inds]
        X_train = embedding[itrain]
        X_test = embedding[itest]
        return {'y_train': y_train, 'y_test': y_test, 'X_train': X_train, 'X_test': X_test}
    # , random_state):

    def train_test_split(self, movie_stim_table, dff_traces, trial, embedding, test_set_size):
        stimuli = movie_stim_table.loc[movie_stim_table['repeat'] == trial]
        X_train, X_test, y_train_inds, y_test_inds = train_test_split(
            embedding, stimuli['start'].values, test_size=test_set_size, random_state=7)  # , random_state=random_state)
        y_train = dff_traces[:, y_train_inds]
        y_test = dff_traces[:, y_test_inds]
        return {'y_train': y_train, 'y_test': y_test, 'X_train': X_train, 'X_test': X_test}

    def regression(self, dat_dct, regression_model):

        metrics = [r2_score]

        y_train, y_test, X_train, X_test = dat_dct['y_train'], dat_dct[
            'y_test'], dat_dct['X_train'], dat_dct['X_test']

        regr = clone(regression_model)
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
        return scores

    def __call__(self, **kwargs: Any) -> Any:
        # train_test_dict = self.train_test_split(
        # kwargs['movie_stim_table'], kwargs['dff_traces'], kwargs['trial'], kwargs['embedding'], kwargs['test_set_size'])  # , random_state)
        # events = kwargs['events']
        train_test_dict = self.train_test_split_interleaved(
            kwargs['movie_stim_table'], kwargs['dff_traces'], kwargs['trial'], kwargs['embedding'], kwargs['test_set_size'])  # , random_state)
        train_test_dict = self.train_test_split_interleaved(
            kwargs['movie_stim_table'], kwargs['events'], kwargs['trial'], kwargs['embedding'], kwargs['test_set_size'])
        scores = self.regression(train_test_dict, kwargs['regression_model'])
        return {'scores': scores}
        # print(train_test_dict)


load_dotenv()

get_events(511511001)

# model = ViTModel.from_pretrained('google/vit-base-patch16-224')
model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

regr_model = Ridge(10)

kwargs = {'model': model,
          'regression_model': regr_model,
          'single_trial_f': MovieSingleTrialRegressionAnalysis_(),
          'test_set_size': 0.25}

t = Test(**kwargs)

print(t)
t(511511001).dropna().scatter_movies().heatmap()

from base import Pipeline
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from pathlib import Path
import os


class NeuronPredictionPipeline(Pipeline):
    def __init__(self, **kwargs):
        self.model = kwargs['model']
        self.regression_model = kwargs['regression_model']
        self.single_trial_f = kwargs['single_trial_f']
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

    def __call__(self, experiment_container_id) -> str:
        output = experiment_container_id
        self.current_experiment_id = experiment_container_id
        self.analysis_callable(self.current_experiment_id)
        print(output)
        return self

    def plot(self, args=None):
        print('plotting')
        return self

    def filter_data(self, args=None):
        print('filtering')
        return self

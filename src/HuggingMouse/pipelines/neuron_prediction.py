from base import Pipeline


class NeuronPredictionPipeline(Pipeline):
    def __init__(self, **kwargs):
        self.model = kwargs['model']
        self.regression_model = kwargs['regression_model']
        self.analysis_function = kwargs['analysis_function']

    def __call__(self, experiment_id) -> str:
        output = experiment_id
        self.current_experiment_id = experiment_id
        print(output)
        return self

    def plot(self, args=None):
        print('plotting')
        return self

    def filter_data(self, args=None):
        print('filtering')
        return self

from neuron_prediction import NeuronPredictionPipeline
from base import Pipeline


def pipeline(task_name: str, **kwargs) -> Pipeline:
    task_mapping = {
        "neuron-prediction": NeuronPredictionPipeline,
    }
    task = task_mapping[task_name]
    return task(**kwargs)
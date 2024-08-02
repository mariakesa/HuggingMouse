from HuggingMouse.pipelines.neuron_prediction import NeuronPredictionPipeline
from HuggingMouse.pipelines.base import Pipeline
import sys
sys.path.append(
    '/home/maria/HuggingMouse/src/HuggingMouse/pipelines/event_prediction.py')
from event_prediction import EventPredictionPipeline


def pipeline(task_name: str, **kwargs) -> Pipeline:
    task_mapping = {
        "neural-activity-prediction": NeuronPredictionPipeline,
        "event-prediction": EventPredictionPipeline,
    }
    task = task_mapping[task_name]
    return task(**kwargs)

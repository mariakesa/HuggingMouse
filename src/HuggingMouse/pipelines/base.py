from abc import abstractmethod


class Pipeline:
    @abstractmethod
    def __call__(self, experiment_id) -> str:
        pass

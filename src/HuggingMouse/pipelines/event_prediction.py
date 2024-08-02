from HuggingMouse.utils import make_container_dict
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from pathlib import Path
import os


class EventPredictionPipeline:
    def __init__(self, **kwargs):
        allen_cache_path = os.environ.get('HGMS_ALLEN_CACHE_PATH')
        if allen_cache_path is None:
            pass
            # raise TransformerEmbeddingCachePathNotSpecifiedError()
        self.boc = BrainObservatoryCache(manifest_file=str(
            Path(allen_cache_path) / Path('brain_observatory_manifest.json')))
        self.eid_dict = make_container_dict(self.boc)
        self.stimulus_session_dict = {
            'three_session_A': ['natural_movie_one', 'natural_movie_three'],
            'three_session_B': ['natural_movie_one'],
            'three_session_C': ['natural_movie_one', 'natural_movie_two'],
            'three_session_C2': ['natural_movie_one', 'natural_movie_two']
        }

    def __call__(self, container_id) -> str:
        self.current_container = container_id
        for session, _ in self.stimulus_session_dict.items():
            try:
                pass
            except Exception as e:
                pass
        return 'Success'

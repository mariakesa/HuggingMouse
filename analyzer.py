from transformers import ViTImageProcessor, ViTModel
from pathlib import Path
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import numpy as np
from get_config_params import get_cache_path
import torch

class MakeEmbeddings:
    cache_path = get_cache_path()
    session_A = 501704220  # This is three session A
    session_B = 501559087
    session_C = 501474098
    boc = BrainObservatoryCache(manifest_file=str(Path(cache_path) / Path('brain_observatory_manifest.json')))
    raw_data_dct = {}
    movie_one_dataset = boc.get_ophys_experiment_data(session_A)
    raw_data_dct['movie_one'] = movie_one_dataset.get_stimulus_template('natural_movie_one')

    def __init__(self, processor, model):
        self.processor = processor
        self.model = model

    def process_stims(self, stims):
        n_stims = len(stims)
        stims = np.repeat(stims[:, np.newaxis, :, :], 3, axis=1)
        embeddings = np.empty((n_stims, 768))
        for i in range(n_stims):
            print(i)
            inputs = self.processor(images=stims[i], return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls = outputs.pooler_output.squeeze().detach().numpy()
            embeddings[i, :] = cls
        return embeddings

    def execute(self):
        embeddings_dct = {}
        for key in self.raw_data_dct.keys():
            print(self.raw_data_dct[key].shape)
            embeddings_dct[key] = self.process_stims(self.raw_data_dct[key])
        return embeddings_dct


if __name__=="__main__":
    CACHE_PATH='/media/maria/DATA/AllenData'
    #Initialize model and processor
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch32-384')
    model = ViTModel.from_pretrained('google/vit-base-patch32-384')
    MakeEmbeddings(processor, model).execute()

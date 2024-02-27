from transformers import ViTImageProcessor, ViTModel
from pathlib import Path
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import numpy as np
from get_config_params import get_cache_paths
import torch
import pickle
import os
from exceptions import CachePathNotSpecifiedError

class MakeEmbeddings:
    allen_cache_path, project_cache_path = get_cache_paths()
    #Experiments where these three types of movies were played
    session_A = 501704220  # This is three session A
    session_B = 501559087
    session_C = 501474098
    boc = BrainObservatoryCache(manifest_file=str(Path(allen_cache_path) / Path('brain_observatory_manifest.json')))
    raw_data_dct = {}
    movie_one_dataset = boc.get_ophys_experiment_data(session_A)
    raw_data_dct['movie_one'] = movie_one_dataset.get_stimulus_template('natural_movie_one')

    def __init__(self, processor, model, cache_data=True):
        self.processor = processor
        self.model = model
        self.cache_data=cache_data


    def process_stims(self, stims):
        n_stims = len(stims)
        n_stims=10
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
    
    def save_to_cache(self, embeddings_dct):
        # Replace / with _ for valid file name
        model_string=self.model.name_or_path.replace('/', '_') 
        file_name = model_string+'_embeddings.pkl'
        # Pickle the dictionary
        with open(file_name, 'wb') as f:
            pickle.dump(embeddings_dct, f)

    def execute(self):
        embeddings_dct = {}
        for key in self.raw_data_dct.keys():
            print(self.raw_data_dct[key].shape)
            embeddings_dct[key] = self.process_stims(self.raw_data_dct[key])
        if self.cache_data:
            if not self.project_cache_path:
                raise CachePathNotSpecifiedError("No project cache path specified in config.json!")
            elif not os.path.exists(self.project_cache_path):
                raise FileNotFoundError(f"Project cache path '{self.project_cache_path}' does not exist!")
            else:
                self.save_to_cache(embeddings_dct)
        return embeddings_dct


if __name__=="__main__":
    CACHE_PATH='/media/maria/DATA/AllenData'
    #Initialize model and processor
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch32-384')
    model = ViTModel.from_pretrained('google/vit-base-patch32-384')
    #model_name_str = model.name_or_path
    #print(model_name_str)
    MakeEmbeddings(processor, model).execute()

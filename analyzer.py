from transformers import ViTImageProcessor, ViTModel
from pathlib import Path
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import numpy as np
from get_config_params import get_cache_paths
import torch
import pickle
import os
from exceptions import CachePathNotSpecifiedError
import pandas as pd

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
    movie_two_dataset = boc.get_ophys_experiment_data(session_C)
    raw_data_dct['movie_two'] = movie_two_dataset.get_stimulus_template('natural_movie_two')
    movie_three_dataset = boc.get_ophys_experiment_data(session_A)
    raw_data_dct['movie_three'] = movie_three_dataset.get_stimulus_template('natural_movie_three')

    def __init__(self, processor, model, cache_data=True):
        self.processor = processor
        self.model = model
        self.cache_data=cache_data


    def process_stims(self, stims):
        n_stims = len(stims)
        #n_stims=10
        stims_dim = np.repeat(stims[:, np.newaxis, :, :], 3, axis=1)
        embeddings = np.empty((n_stims, 768))
        for i in range(n_stims):
            print(i)
            inputs = self.processor(images=stims_dim[i], return_tensors="pt")
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
        save_path=Path(self.project_cache_path) / Path(file_name)
        with open(save_path, 'wb') as f:
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

class AllenExperimentUtility:
    def __init__(self):
        allen_cache_path, project_cache_path = get_cache_paths()
        boc = BrainObservatoryCache(manifest_file=str(Path(allen_cache_path) / 'brain_observatory_manifest.json'))
        
    def view_all_imaged_areas(self):
        print(self.boc.get_all_targeted_structures())

    def view_all_cre_lines(self):
        print(self.boc.get_all_cre_lines())

    def experiment_container_ids_imaged_areas(self, imaged_areas):
        experiment_containers=self.boc.get_experiment_containers(imaged_areas=imaged_areas)
        print('These are experimental id\'s that contain query imaged areas: ',
              self.boc.get_experiment_containers(imaged_areas=imaged_areas))
        return [exp_c['id'] for exp_c in experiment_containers]
    

    
class PreprocessNeuralData:
    def __init__(self):
        self.allen_cache_path, self.project_cache_path = get_cache_paths()
        self.boc = BrainObservatoryCache(manifest_file=str(Path(self.allen_cache_path) / 'brain_observatory_manifest.json'))
        self.eid_dict = self.make_container_dict()
    
    def make_container_dict(self):
        '''
        This method parses which experimental id's (values)
        correspond to which experiment containers (keys).
        '''
        experiment_container = self.boc.get_experiment_containers()
        container_ids = [dct['id'] for dct in experiment_container]
        eids = self.boc.get_ophys_experiments(experiment_container_ids=container_ids)
        df = pd.DataFrame(eids)
        reduced_df = df[['id', 'experiment_container_id', 'session_type']]
        grouped_df = reduced_df.groupby(['experiment_container_id', 'session_type'])['id'].agg(list).reset_index()
        eid_dict = {}
        for row in grouped_df.itertuples(index=False):
            container_id, session_type, ids = row
            if container_id not in eid_dict:
                eid_dict[container_id] = {}
            eid_dict[container_id][session_type] = ids[0]
        return eid_dict


if __name__=="__main__":
    #Initialize model and processor
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch32-384')
    model = ViTModel.from_pretrained('google/vit-base-patch32-384')
    #model_name_str = model.name_or_path
    #print(model_name_str)
    #MakeEmbeddings(processor, model).execute()
    boc = BrainObservatoryCache(manifest_file=str(Path("/media/maria/DATA/AllenData") / 'brain_observatory_manifest.json'))
    experiment_container = boc.get_experiment_containers()
    print(experiment_container)

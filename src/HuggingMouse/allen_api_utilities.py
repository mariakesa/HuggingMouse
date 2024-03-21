from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from pathlib import Path
import os
from HuggingMouse.custom_exceptions import AllenCachePathNotSpecifiedError, TransformerEmbeddingCachePathNotSpecifiedError


class AllenExperimentUtility:
    '''
    Class for printing out container id's of experiments and
    information related to the transgenic lines and visual areas.
    It's meant to be useful for selecting an experiment container
    with neural activity to fit image-derived features to. 
    '''

    def __init__(self):
        allen_cache_path = os.environ.get('HGMS_ALLEN_CACHE_PATH')
        if allen_cache_path is None:
            raise AllenCachePathNotSpecifiedError()
        transformer_embedding_cache_path = os.environ.get(
            'HGMS_TRANSF_EMBEDDING_PATH')
        if transformer_embedding_cache_path is None:
            raise TransformerEmbeddingCachePathNotSpecifiedError()
        self.boc = BrainObservatoryCache(manifest_file=str(
            Path(allen_cache_path) / 'brain_observatory_manifest.json'))

    def view_all_imaged_areas(self):
        print(self.boc.get_all_targeted_structures())

    def view_all_cre_lines(self):
        print(self.boc.get_all_cre_lines())

    def cre_line_info(self):
        print(
            '''
            Transgenic characterizations can be found at https://observatory.brain-map.org/visualcoding/transgenic
            '''
        )

    def visual_areas_info(self):
        print(
            '''
        Mouse visual areas in Allen Brain Observatory: VISal, VISam, VISl, VISp (V1), VISpm, VISrl
        From "Functional Specialization of Seven Mouse Visual Cortical Areas", Marshel et al, Neuron 2012:
        Two major parallel processing pathways have been defined based on functional specializations, patterns 
        of connections, and associations with different behaviors. The dorsal pathway is specialized to process motion 
        and spatial relationships and is related to behaviors involving visually guided actions. The ventral pathway is 
        specialized to process fine-scale detail, shapes and patterns in an image to support object recognition and is 
        associated with visual perception.
        Each of the six mouse extrastriate visual areas we investigated contains neurons that are 
        highly selective for fundamental visual features, including orientation, direction, spatial 
        frequency (SF) and temporal frequency (TF).
        All extrastriate areas investigated, with the exception of PM, encode faster TFs than VISp, suggesting a role 
        for these higher areas in the processing of visual motion. For a subset of areas, VISal, VISrl and VISam, this role is 
        further supported by a significant increase in direction selectivity across each population. Another subset of 
        areas, VISli (not in Allen data) and VISpm, prefer high SFs, suggesting a role in the processing of structural detail in an 
        image. Nearly all higher visual areas improve orientation selectivity compared to VISp.
        Areas VISal, VISrl and VISam are all highly direction selective and respond to high TFs and low SFs. These properties have 
        served as hallmarks of the dorsal pathway in other species, and suggest that AL, RL and AM perform 
        computations related to the analysis of visual motion.In contrast, areas VISli(not in Allen data) and VISpm respond to high SFs, 
        and PM is highly orientation selective, suggesting a role in the analysis of structural detail and form in an image.
        These results suggest that the mouse visual cortex may be organized into groups of specialized areas that process information 
        related to motion and behavioral actions versus image detail and object perception, analogous to the dorsal and 
        ventral streams described in other species.
        ''')

    def experiment_container_ids_imaged_areas(self, imaged_areas):
        experiment_containers = self.boc.get_experiment_containers(
            targeted_structures=imaged_areas)
        ecids = [exp_c['id'] for exp_c in experiment_containers]
        print('These are experimental containers\'s that contain query imaged areas: ',
              experiment_containers)
        print('These are experimental container id\'s corresponding to imaged areas', ecids)
        return ecids

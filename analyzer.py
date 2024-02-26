from transformers import ViTImageProcessor, ViTModel
from pathlib import Path
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

class MakeEmbeddings:
    def __init__(self, processor, model,cache_path):
        self.boc = BrainObservatoryCache(manifest_file=str(
            Path(cache_path) / Path('brain_observatory_manifest.json')))

if __name__=="__main__":
    CACHE_PATH='/media/maria/DATA/AllenData'
    #Initialize model and processor
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch32-384')
    model = ViTModel.from_pretrained('google/vit-base-patch32-384')
    MakeEmbeddings(processor, model, CACHE_PATH)
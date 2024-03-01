import json
from exceptions import CachePathNotSpecifiedError
import os
from pathlib import Path

def create_config_file():
    config = {
        "allen_cache_path": str(Path(os.path.dirname(__file__))/Path('AllenCache')),
        "transformer_embedding_cache_path": str(Path(os.path.dirname(__file__))/Path('TransformerEmbeddings'))
    }
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def get_cache_paths(): 
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'config.json')):
        create_config_file()
    # Opening JSON 
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    f = open(config_path)
    config = json.load(f)

    if not config['allen_cache_path']:
        CachePathNotSpecifiedError("No Allen data cache path specified in config.json!")
    #Different logic for two caches-- project_cache is optional, but there's no way around
    #Allen cache
    return config['allen_cache_path'], config.get('transformer_embedding_cache_path', None)
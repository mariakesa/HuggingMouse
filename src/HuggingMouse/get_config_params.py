import json
from exceptions import CachePathNotSpecifiedError

def get_cache_paths(): 
    # Opening JSON file
    f = open('/home/maria/HuggingMouse/src/HuggingMouse/config.json')
    config = json.load(f)

    if not config['allen_cache_path']:
        CachePathNotSpecifiedError("No Allen data cache path specified in config.json!")
    #Different logic for two caches-- project_cache is optional, but there's no way around
    #Allen cache
    return config['allen_cache_path'], config.get('transformer_embedding_cache_path', None)
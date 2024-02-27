import json

def get_cache_paths(): 
    # Opening JSON file
    f = open('/home/maria/HuggingMouse/config.json')
    
    # returns JSON object as 
    # a dictionary
    config = json.load(f)

    return config['allen_cache_path'], config.get('project_cache_path', None)
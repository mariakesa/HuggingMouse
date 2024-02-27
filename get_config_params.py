import json

def get_cache_path(): 
    # Opening JSON file
    f = open('/home/maria/HuggingMouse/config.json')
    
    # returns JSON object as 
    # a dictionary
    config = json.load(f)

    return config['cache_path']
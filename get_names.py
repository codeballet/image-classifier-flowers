# Dictionary mapping category label to category name
import json
    
    
def cat_to_name(filepath):
    with open(filepath, 'r') as f:
        cat_dict = json.load(f)
        
    return cat_dict
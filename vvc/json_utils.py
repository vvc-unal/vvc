import jsonpickle
import json

import jsonpickle.ext.numpy as jsonpickle_numpy


jsonpickle_numpy.register_handlers()


def save_to_json(data, filename):
    
    # Serialize object
    json_obj = jsonpickle.encode(data, unpicklable=False)
    
    #json.load method converts JSON string to Python Object
    parsed = json.loads(json_obj)
    
    # Save data to file
    with open(filename, "w") as write_file:
        json.dump(parsed, write_file, indent=2)

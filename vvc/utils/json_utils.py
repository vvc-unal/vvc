import json

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy


jsonpickle_numpy.register_handlers()


def save_to_json(data, filename):
    
    # Serialize object
    json_obj = jsonpickle.encode(data, unpicklable=True)
    
    # json.load method converts JSON string to Python Object
    parsed = json.loads(json_obj)
    
    # Save data to file
    with open(filename, "w") as write_file:
        json.dump(parsed, write_file, indent=2, sort_keys=True)


def load_from_json(json_file):
    
    with open(json_file, 'r') as f:
        data = f.read()
        # Deserialize object
        return jsonpickle.decode(str(data))


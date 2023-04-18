import json
import os


def get_model_params(parameters_dir):
    params = {}
    for file in os.listdir(f"{parameters_dir}"):
        if file.endswith(".json"):
            with open(f"{parameters_dir}/{file}") as f:
                params[os.path.splitext(file)[0]] = json.load(f)

    return params

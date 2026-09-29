import os
import yaml
import json

def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    return content

def create_directory(dirs: list):
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def save_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
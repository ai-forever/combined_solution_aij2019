import json
import os
import pickle


def load_tasks(dir_path, task_num=None):
    tasks, filenames = [], [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
    for filename in filenames:
        if filename.endswith(".json"):
            with open(filename, encoding="utf-8") as f:
                dt = f.read().encode("utf-8")
                data = json.loads(dt)
                tasks += [d for d in data if "id" in d and int(d["id"]) == task_num]
    return tasks


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def read_config(config_path):
    if isinstance(config_path, str):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    return config


def save_config(config_path, object_to_save):
    if isinstance(config_path, str):
        with open(config_path, "w+", encoding="utf-8") as f:
            json.dump(object_to_save, f, ensure_ascii=False, indent=4)

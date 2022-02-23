import pandas as pd
from pathlib import Path
import os
from utils import command_eng, command_ita
from global_utils import select_analyzator, EXCLUDES

curr_dir = os.path.abspath(os.path.dirname(os.path.relpath(__file__)))
path_base = Path(f"{curr_dir}/dataset/Dataset_synth")

keys = ["#speaker_ita", "#speaker_eng", "#cmds_ita", "#cmds_eng", "service_name"]

def init_dict() -> dict:
    data = {}
    for key in keys:
        data[key] = 0
    return data

def print_data(data: dict, db_eng: pd.DataFrame, db_ita: pd.DataFrame, service_name):
    filpath = path_base.joinpath(service_name, "info.csv")
    out = {}
    for k in keys:
        out[k] = [data[k]]
    pd.DataFrame(out).to_csv(filpath)
    db_eng.to_csv(path_base.joinpath(service_name, "db_eng.csv"))
    db_ita.to_csv(path_base.joinpath(service_name, "db_ita.csv"))

def count_decorator(func):
    def wrapper(service_name):
        db_eng = pd.DataFrame(data=[0]*len(keys_eng), index=keys_eng, columns=["eng"])
        db_ita = pd.DataFrame(data=[0]*len(keys_ita), index=keys_ita, columns=["ita"])
        data, db_eng, db_ita = func(db_eng, db_ita, service_name)
        print_data(data, db_eng, db_ita, service_name)
    return wrapper

@count_decorator
def count(db_eng: pd.DataFrame, db_ita: pd.DataFrame, service_name):
    data = init_dict()
    voices = {"eng": set(), "ita": set()}
    path = path_base.joinpath(service_name)
    analyze_str = select_analyzator(service_name)
    for root, dirs, files in os.walk(path, followlinks=True, topdown=False):
        if len(files) == 0:
            continue
        for fil in files:
            if fil in EXCLUDES: continue
            folder_lang = Path(root).name
            db = db_eng if folder_lang == "eng" else db_ita
            voice, cmd_index = analyze_str(fil)
            voices[folder_lang].add(voice)
            db.iat[cmd_index, 0] += 1

    data[keys[0]] = len(voices["ita"])
    data[keys[1]] = len(voices["eng"])
    data[keys[2]] = db_ita["ita"].sum()
    data[keys[3]] = db_eng["eng"].sum()
    data[keys[4]] = service_name
    return data, db_eng, db_ita

def init_keys():
    keys_eng = []
    keys_ita = []
    for index, key in command_eng.items():
        keys_eng.append(key)
    for index, key in command_ita.items():
        keys_ita.append(key)
    return keys_eng, keys_ita

if __name__ == "__main__":
    keys_eng, keys_ita = init_keys()
    count("azure")
    count("google")
    count("ibm")
    count("nemo")
    count("polly")
    count("vocalware")
    count("naturaltts")
    count("naturalreaders")
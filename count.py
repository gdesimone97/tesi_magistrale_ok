import os
from pathlib import Path

import pandas
import pandas as pd
import matplotlib.pyplot as plt
from utils import command_ita, command_eng

def get_curr_dir(file):
    path = os.path.abspath(os.path.realpath(os.path.dirname(file)))
    return path

def plot(lang):
    assert lang == "ita" or lang == "eng"
    command_list = command_ita if lang == "ita" else command_eng
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(17, 6))
    commands = list(range(0, 26))
    data = []

    for e in commands:
        count = 0
        for folder in os.listdir(path):
            target = path.joinpath(folder,lang, f"{lang}_{str(e)}.ogg")
            if target.exists():
                count += 1
        data.append(count)

    command_list = list(command_list.values())
    ax.barh(command_list, data, align='center')
    ax.set_yticks(commands)
    ax.set_yticklabels(command_list)
    ax.set_title(lang)
    ax.tick_params(axis='y', which='major', labelsize=7)
    plt.xlabel("#samples")
    plt.ylabel("command")
    for i, e in enumerate(data):
        plt.text(e + 0.01, commands[i] - 0.25 , str(e))

    path_save = Path(get_curr_dir(__file__), "figure")
    path_save.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_save.joinpath(f"{lang}.png"))
    pd.DataFrame(data, index=command_list, columns=[lang]).to_csv(path_save.joinpath(f"{lang}.csv"))
    data = {"#speaker": [count_speaker], "#cmds_eng": [count_eng], "#cmds_ita": [count_ita], "#cmds_tot": [count_tot]}
    pd.DataFrame(data).to_csv(path_save.joinpath("info.csv"))
    #plt.show()


path = Path(f"{get_curr_dir(__file__)}/saves/")

count_ita = 0
count_eng = 0
count_speaker = 0
for folder in os.listdir(path):
    path_db = path.joinpath(folder, "database.csv")
    database = pd.read_csv(path_db, index_col=0)
    count_eng += len(database.loc[database["done_eng"] == True])
    count_ita += len(database.loc[database["done_ita"] == True])
    if path.joinpath(folder, "eng").exists() or path.joinpath(folder, "ita").exists():
        count_speaker += 1

count_tot = count_ita + count_eng

print("#English samples:", count_eng)
print("#Italian samples:", count_ita)
print("#Total:", count_tot)
print("#Speakers:", count_speaker)

plot("eng")
plot("ita")
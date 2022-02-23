import os

import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

def plot_reset():
    plt.clf()

def plot(df: pd.DataFrame, fname, title="", xlabel="", reset=True):
    x = df["Step"]
    y = df["Value"]
    plt.xlabel("Epoch")
    plt.title(title)
    plt.plot(x, y, label=xlabel)
    plt.legend(loc="best")
    fname = fname.split(".")[0] + ".jpg"
    out_path = fig_folder.joinpath(fname)
    plt.savefig(out_path)
    if reset:
        plot_reset()

path = r"C:\MIE CARTELLE\PROGRAMMAZIONE\GITHUB\tesi_magistrale\res\eng"
folder_path = Path(path)
fig_folder = folder_path.joinpath("fig")
fig_folder.mkdir(exist_ok=True)

path = folder_path.joinpath("train_loss.csv")
df = pd.read_csv(path)
del df["Wall time"]
df_out = df.iloc[::10]
df_out: pd.DataFrame = df_out.append(df.iloc[-1])
df_out["Step"] = df_out.reset_index().index
df_out["Step"] = df_out["Step"].astype("int32")
path_out = Path(path).parent.joinpath("train_loss_sample.csv")
df_out.to_csv(str(path_out), index=False)
plot(df_out, path_out.name, title="train_loss", reset=False)

path = folder_path.joinpath("val_loss.csv")
df = pd.read_csv(path)
del df["Wall time"]
df_out = df.iloc[:]
df_out["Step"] = df_out.reset_index().index
df_out["Step"] = df_out["Step"].astype("int32")
path_out = Path(path).parent.joinpath("val_loss_sample.csv")
df_out.to_csv(str(path_out), index=False)
plot(df_out, path_out.name, title="validation_loss")

EXCLUDE = ["train_loss_sample.csv", "val_loss_sample.csv", "stats.csv"]

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file in EXCLUDE: continue
        title = file.split(".")[0]
        df = pd.read_csv(Path(root).joinpath(file))
        plot(df, fname=file, title=title, xlabel=title)
    break
plot_reset()
for root, dirs, files in os.walk(folder_path):
    files_list = []
    for file in files:
        if file in EXCLUDE: continue
        if file == "acc_mean.csv": continue
        if file.split("_")[0] != "acc": continue
        files_list.append(file)
    files_list.sort(key=lambda x: int(x.split("_")[1].split("db")[0]))
    for file in files_list:
        title = "Accuracy on validation set"
        df = pd.read_csv(Path(root).joinpath(file))
        fname = "acc.csv"
        xlabel = file.split(".")[0]
        plot(df, fname=fname, title=title, xlabel=xlabel, reset=False)
    break
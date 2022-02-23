import os
import shutil

import pandas as pd
from pathlib import Path
import torch
from model import Model, ValidationCallback
from global_utils import get_curr_dir
from omegaconf import OmegaConf
import omegaconf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
import commands
from plot_tf_event import PlotManager
import pickle
from utils import is_windows
# from pytimedinput import timedInput
import argparse

def save_predicts(predict_labels: dict, true_labels: dict):
    data = (predict_labels, true_labels)
    with open(str(predict_file_path), "wb") as fil:
        pickle.dump(data, fil)

def load_predict() -> np.ndarray:
    with open(str(predict_file_path), "rb") as fil:
        return pickle.load(fil)

def reject_acc(y_true, y_pred):
    assert len(commands.command_eng) == len(commands.command_ita)
    y_true = np.where(y_true == len(commands.command_eng), 0, 1)
    y_pred = np.where(y_pred == len(commands.command_eng), 0, 1)
    return accuracy_score(y_true, y_pred)

def predict_sigmoid(logits, limit=0.0):
    sigmoid = torch.nn.Sigmoid()
    predict_labels = []
    for logits_row in logits:
        out_list = []
        for logit in logits_row:
            out = sigmoid(logit).item()
            out_list.append(out)
        out_list = np.array(out_list)
        mask: np.ndarray = out_list >= limit
        if not mask.any():
            predict_labels.append(len(commands.command_eng))
        else:
            predict_labels.append(np.argmax(out_list))
    return np.array(predict_labels)

def report(true_labels, predict_labels, snr, seed):
    report = classification_report(true_labels, predict_labels, labels=labels)
    path = Path(exp_dir).joinpath(f"res/report_{snr}dB.txt")
    fil = open(str(path), "a")
    print(f"\t{ckpt_name}\n\tsnr: {snr} - seed: {seed}\n{report}", file=fil)
    fil.close()

def metrics(true_labels, predict_labels, seed, snr):
    acc = accuracy_score(y_true=true_labels, y_pred=predict_labels)
    rej_acc = reject_acc(true_labels, predict_labels)
    bal_acc = balanced_accuracy_score(y_true=true_labels, y_pred=predict_labels)
    stat_data = {"seed": int(seed), "snr": int(snr), "accuracy": acc, "bal_acc": bal_acc, "rejcet_acc": rej_acc}
    return stat_data

def is_test():
    if predict_file_path.exists():
        # userText, timedOut = timedInput("Do you want re-test? [y/n]:")
        res = input("Do you want re-test? [y/n]:")
        if res.lower() == "y":
            pass
        else:
            return False
    res_path = Path(exp_dir).joinpath("res")
    if res_path.exists():
        try:
            shutil.rmtree(res_path)
        except OSError:
            pass
    res_path.mkdir(parents=True, exist_ok=True)
    return True

if __name__ == "__main__":
    seed_list  = [0]
    SNR_MIN = 0
    SNR_MAX = 30
    SNR_STEP = 5
    NUM_CPU = 6
    snr_range = range(SNR_MIN, SNR_MAX+SNR_STEP, SNR_STEP)
    working_dir = Path(get_curr_dir(__file__)).joinpath(".working_dir/test")
    stats = pd.DataFrame(columns=["seed", "snr", "accuracy", "bal_acc", "rejcet_acc"])
    if not is_windows():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(3)
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=False, default=None, dest="exp_dir")
    parser.add_argument("--ckpt", type=str, required=False, default=None, dest="ckpt")
    args = parser.parse_args()
    if args.exp_dir is None:
        exp_dir = r"C:\MIE CARTELLE\PROGRAMMAZIONE\GITHUB\tesi_magistrale\nemo_experiments\MatchboxNet-3x2x64\2022-01-19_19-42-14"
    else:
        exp_dir = args.exp_dir
    if not is_windows():
        exp_dir = Path(exp_dir).as_posix()
    if args.ckpt is None:
        ckpt_name = r"matchcboxnet--val_loss=0.492-epoch=145.model"
    else:
        ckpt_name  = args.ckpt
    ckpt_name = str(Path(ckpt_name).name)
    cfg_path = Path(exp_dir).joinpath("hparams.yaml")
    print("*"*50)
    print("exp_dir:", exp_dir)
    print("ckpt_name:", ckpt_name)
    print("*"*50)
    config = OmegaConf.load(cfg_path)
    config = OmegaConf.create(config)
    lang = config.cfg.lang
    test_manifest_path = Path(exp_dir).joinpath(rf"db/test_{lang}_manifest.csv")
    noise_test_path = Path(exp_dir).joinpath(rf"db/noise_test_{lang}_manifest.csv")
    # ckpt = Path(exp_dir).joinpath("checkpoints", ckpt_name)

    predict_file_path = Path(exp_dir).joinpath("res/predict.data")

    try:
        SR = config.sample_rate
    except omegaconf.errors.ConfigAttributeError:
        config = config.cfg
        SR = config.sample_rate

    # asr_model: Model = Model.load_from_checkpoint(str(ckpt), last_layer="softmax")
    asr_model = Model.load_backup(ckpt_name, exp_dir)
    batch_size = config["train_ds"]["batch_size"]
    sample_rate = config["sample_rate"]
    labels = config["labels"]
    max_gain_db = config.train_ds.augmentor.mynoise.max_gain_db

    test = is_test()
    plot_manager = PlotManager(exp_dir, SNR_MIN, SNR_MAX, SNR_STEP)
    plot_manager.plot_all()
    for i, seed in enumerate(seed_list):
        validation_callback = ValidationCallback(labels, batch_size, snr_range,
                                                 sample_rate, lang, seed, max_gain_db,
                                                 noise_test_path, test_manifest_path, test=True)
        dataloaders = validation_callback.dataloaders
        predict_db, true_labels_db = {}, {}
        for snr, dataloader in dataloaders.items():
            if test:
                try:
                    _, true_labels, predict_labels, logtis = validation_callback.validate(asr_model, dataloader, device="gpu")
                except RuntimeError:
                    _, true_labels, predict_labels, logtis = validation_callback.validate(asr_model, dataloader, device="cpu")
                predict_db[snr] = predict_labels
                true_labels_db[snr] = true_labels
                save_predicts(predict_db, true_labels_db)
            else:
                predict_db, true_labels_db = load_predict()
                predict_labels, true_labels = predict_db[snr], true_labels_db[snr]
            stat_data = metrics(true_labels, predict_labels, seed=seed, snr=snr)
            print(stat_data)
            report(true_labels, predict_labels, snr, seed)
            if i == 0:
                matrix = confusion_matrix(true_labels, y_pred=predict_labels, labels=labels)
                disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
                disp.plot()
                plt.title(f"snr: {snr} - seed: {seed}")
                plot_manager.save_confusion_matrix(plt.gcf(), snr)
            stats = stats.append(stat_data, ignore_index=True)
        stats.sort_values(by=["snr", "seed"], inplace=True)
        print(stats)
    print(stats)
    out = Path(get_curr_dir(__file__)).joinpath(f"{exp_dir}/res")
    out.mkdir(exist_ok=True)
    out = str(out.joinpath("stats.csv"))
    print("Stats saved to", out)
    stats.to_csv(out, index=False)
import numpy as np
import pandas as pd
from pathlib import Path
import os
from model import Model, ValidationDataset
from global_utils import get_curr_dir
from omegaconf import OmegaConf
import torch
import json
from utils import extract_logits
import pickle
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, PrecisionRecallDisplay, precision_recall_curve
import matplotlib.pyplot as plt

FIX_MODEL = {
    "eng": {"exp_dir": str(Path(get_curr_dir(__file__)).joinpath("nemo_experiments/MatchboxNet-3x2x64/2022-01-19_23-29-46")), "ckpt": "matchcboxnet--val_loss=0.369-epoch=249.model"},
    "ita": {"exp_dir": str(Path(get_curr_dir(__file__)).joinpath("nemo_experiments/MatchboxNet-3x2x64/2022-01-31_17-32-48")), "ckpt": "matchcboxnet--val_loss=0.3033-epoch=220.model"}
}

def get_validation_path(exp_dir, snr="all"):
    db = pd.DataFrame(columns=["path", "fname"])
    if snr == "all":
        validation_path = Path(exp_dir).joinpath("db/clips")
    else:
        validation_path = Path(exp_dir).joinpath(f"db/clips/{snr}")
    for root, dirs, files in os.walk(validation_path):
        for fil in files:
            fil_path = Path(root).joinpath(fil)
            db = db.append({"path": fil_path, "fname": fil}, ignore_index=True)
    return db

def get_model(exp_dir, ckpt):
    return Model.load_backup(ckpt_name=ckpt, exp_dir=exp_dir)

def save_probs(probs, true_labels, save_path):
    obj = (probs, true_labels)
    with open(save_path, "wb") as fil:
        pickle.dump(obj, fil)

def load_probs(save_path):
    with open(save_path, "rb") as fil:
        probs, true_labels = pickle.load(fil)
        return probs, true_labels

def pandas2manifest(db: pd.DataFrame, exp_dir):
    fil_path = Path(exp_dir).joinpath("roc/roc_manifest.json")
    fil = open(fil_path, "w")
    start_db = pd.read_csv(Path(exp_dir).joinpath(f"db/validation_{lang}_manifest.csv"))
    fname_list, speaker_lsit = [], []
    for e in start_db["path"].values:
        fname_list.append(Path(e).name)
    start_db["fname"] = fname_list
    for index, row in db.iterrows():
        path = row["path"]
        temp = Path(row["fname"]).name
        fname = ""
        for index, e in enumerate(temp.split("_")):
            if index == 0:
                speaker = e
                continue
            fname += e + "_"
        fname = fname[:-1]
        value: pd.Series = start_db.loc[(start_db["speaker_id"] == speaker) & (start_db["fname"] == fname)]
        assert len(value) == 1
        cmd = value["cmd_index"].values[0]
        data = {"audio_filepath": str(path), "duration": 0.0, "command": int(cmd)}
        json.dump(data, fil)
        fil.write('\n')
    fil.close()
    return fil_path

def predict_probs(model, exp_dir, db, snr: str):
    save_path = Path(exp_dir).joinpath(f"roc/{snr}/predict.data")
    roc_folder = Path(exp_dir).joinpath(f"roc/{snr}")
    roc_folder.mkdir(parents=True, exist_ok=True)
    if not save_path.exists():
        assert len(db) != 0
        cfg_path = Path(exp_dir).joinpath("hparams.yaml")
        config = OmegaConf.load(cfg_path)
        config = OmegaConf.create(config)["cfg"]
        batch_size = 1 #config["train_ds"]["batch_size"]
        sample_rate = config["sample_rate"]
        labels = config["labels"]
        manifest_path = pandas2manifest(db, exp_dir)
        dataloader = ValidationDataset(sample_rate=sample_rate, labels=labels).get_dataloader(manifest_path, batch_size)
        logits, true_labels = extract_logits(model, dataloader, device="cpu")
        probs = torch.nn.functional.softmax(logits, dim=-1)
        save_probs(probs, true_labels, save_path)
    else:
        probs, true_labels = load_probs(save_path)
    return probs.cpu().detach().numpy(), true_labels.cpu().detach().numpy()

def verify(true_labels):
    data = pd.DataFrame({"cmd": true_labels}).value_counts()
    data = data.sort_index()
    print(data)

def curve(probs, true_labels):
    REJECT_LABEL = probs.shape[1] - 1
    y = np.where(true_labels == REJECT_LABEL, 0, 1)
    reject_probs = np.take(probs, REJECT_LABEL, axis=1)
    fpr, tpr, thresholds = roc_curve(y, reject_probs, pos_label=0, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc

def precision(probs, true_labels):
    REJECT_LABEL = probs.shape[1] - 1
    th_range = np.linspace(0.0, 1, 11)
    for th in th_range:
        pred_labels = []
        for e in probs:
            if e[REJECT_LABEL] > th:
                pred_labels.append(REJECT_LABEL)
            else:
                e = e[:REJECT_LABEL]
                label = np.argmax(e)
                pred_labels.append(label)
        report = classification_report(true_labels, pred_labels)
        print("*"*30)
        print(f"Report - threshold: {th}")
        print(report)
        print("*"*30)

def compute_thresholds(fpr, tpr, thresholds):
    FPR_LIMIT = 0.05
    # thresholds = thresholds[::-1]
    th_target = np.NAN
    tpr_max = -1
    for th, tpr_v, fpr_v in zip(thresholds, tpr, fpr):
        if fpr_v < FPR_LIMIT and tpr_v > tpr_max:
            th_target = th
            tpr_max = tpr_v
    print("TPR MAX:", tpr_max, "\nThreshold:", th_target)

def precision_recall(probs, true_labels):

    def compute(precison, recall, thresholds):
        print("*"*30)
        print("Precision-recall:\n")
        TARGET_PRECISION = 0.9
        TARGET_RECALL = 0.9
        th_target = np.NAN
        prec_target = -1
        rec_target = -1
        for prec, rec, th in zip(precison, recall, thresholds):
            if prec >= TARGET_PRECISION and rec >= TARGET_RECALL:
                th_target = th
                prec_target = prec
                rec_target = rec
        print("Rec target:", rec_target, "\nPrecision target:", prec_target, "\nThreshold:", th_target)

    REJECT_LABEL = probs.shape[1] - 1
    y_true = np.where(true_labels == REJECT_LABEL, 1, 0)
    reject_probs = np.take(probs, REJECT_LABEL, axis=1)
    indices = range(REJECT_LABEL)
    commands_probs = np.take(probs, indices, axis=1)
    pred_labels = np.amax(commands_probs, axis=1)

    precision_rejcet, recall_reject, thresholds_reject = precision_recall_curve(y_true, reject_probs, pos_label=1)
    # print(precision_rejcet)
    # print(recall_reject)
    # print(thresholds_reject)
    display = PrecisionRecallDisplay.from_predictions(y_true, reject_probs, name="Reject", pos_label=1)
    _ = display.ax_.set_title("Reject")
    plt.show()
    compute(precision_rejcet, recall_reject, thresholds_reject)
    print()

    y_true = np.where(true_labels == REJECT_LABEL, 1, 0)
    precision_cmd, recall_cmd, thresholds_cmd = precision_recall_curve(y_true, pred_labels, pos_label=0)
    # print(precision_cmd)
    # print(recall_cmd)
    # print(thresholds_cmd)
    display = PrecisionRecallDisplay.from_predictions(y_true, pred_labels, name="Command", pos_label=0)
    _ = display.ax_.set_title("Command")
    compute(precision_cmd, recall_cmd, thresholds_cmd)
    plt.show()
    print()


def plot(fpr, tpr, thresholds, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.3f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC for {lang.upper()}")
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    lang = "ita"

    snr = "0"
    exp_dir = FIX_MODEL[lang]["exp_dir"]
    db = get_validation_path(exp_dir, snr)
    model = get_model(exp_dir, FIX_MODEL[lang]["ckpt"])
    model = model.eval()
    probs, true_labels = predict_probs(model, exp_dir, db, snr)
    verify(true_labels)
    fpr, tpr, thresholds, roc_auc = curve(probs, true_labels)
    compute_thresholds(fpr, tpr, thresholds)
    # plot(fpr, tpr, thresholds, roc_auc)
    # precision(probs, true_labels)
    precision_recall(probs, true_labels)

import platform
import string
import random
from pathlib import Path
from global_utils import get_curr_dir
import sys
import pandas as pd
import torch
import shutil, os

class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

def split_for_parallel(df: pd.DataFrame, num_cpu: int):
    db_list = []
    interval = len(df) // num_cpu
    for i in range(num_cpu):
        start = i*interval
        end = i*interval + interval
        if i == num_cpu-1:
            db_split = df.iloc[start:]
        else:
            db_split = df.iloc[start:end]
        db_list.append(db_split)
    return db_list

def extract_logits(model, dataloader, device="cpu"):
    logits_buffer = []
    label_buffer = []

    # Follow the above definition of the test_step
    for batch in dataloader:
        audio_signal, audio_signal_len, labels, labels_len = batch
        if device == "gpu":
            model = model.cuda()
            audio_signal, audio_signal_len, labels = audio_signal.cuda(), audio_signal_len.cuda(), labels.cuda()
        elif device == "cpu":
            model = model.cpu()
            audio_signal, audio_signal_len, labels = audio_signal.cpu(), audio_signal_len.cpu(), labels.cpu()
        logits = model(input_signal=audio_signal, input_signal_length=audio_signal_len)

        logits_buffer.append(logits)
        label_buffer.append(labels)

    logits = torch.cat(logits_buffer, 0)
    labels = torch.cat(label_buffer, 0)
    return logits, labels

def is_windows():
    return platform.system().lower() == "windows"

def get_random_string(length) -> str:
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


class Logger:
    def __init__(self):
        self._exp_dir = None
        while True:
            temp_file = Path(get_curr_dir(__file__)).joinpath(self._get_temp_path())
            if not temp_file.exists(): break
        self.temp_file = temp_file
        sys.stdout = open(str(temp_file), "w")
        sys.stderr = sys.stdout

    def _get_temp_path(self):
        temp_name = get_random_string(10) + ".txt"
        return Path(get_curr_dir(__file__)).joinpath(temp_name)

    @property
    def exp_dir(self):
        return self._exp_dir

    @exp_dir.setter
    def exp_dir(self, exp_dir):
        self._exp_dir = Path(exp_dir)
        sys.stdout.close()
        self.logfile_path = self._exp_dir.joinpath("log.txt")
        shutil.copy(self.temp_file, self.logfile_path)
        os.remove(self.temp_file)
        self.logfile = open(self.logfile_path, "a")
        sys.stdout = self.logfile
        sys.stderr = sys.stdout

    def close(self):
        sys.stderr.close()
        sys.stdout.close()
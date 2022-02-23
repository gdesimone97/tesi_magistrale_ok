from pathlib import Path
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from utils import Logger
from global_utils import get_curr_dir
from commands import command_eng, command_ita
from model import Augmentation, ValidationCallback, Model
import subprocess
import platform
import os
import argparse

try:
    from google.colab import drive
    COLAB = True
except Exception:
    COLAB = False


if __name__ == "__main__":
    '''
    Parsing input argument
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, dest="config", required=True, help="Yaml file containing the configuration")
    parser.add_argument("--lang", type=str, dest="lang", required=False, default="eng", help='Desired lang')
    parser.add_argument("--id", type=str, required=True, dest="train_id", help='training id')
    parser.add_argument("--log", type=int, required=False, dest="logfile", default=1, help='0 to disable log file, 1 to activate log file, default 1')
    parser.add_argument("--synth", type=int, required=False, dest="synth", default=1, help='0 to exculde synthetic samples, 1 to include it, default 1')
    args = parser.parse_args()

    if args.logfile:
        logger = Logger()

    from nemo.collections.asr.parts.preprocessing.perturb import register_perturbation
    from nemo.utils.exp_manager import exp_manager

    print("*"*30)
    print("Training id", args.train_id)
    print("*" * 30)

    if not COLAB and platform.system().lower() != "windows":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(3)
    register_perturbation("mynoise", Augmentation)
    path_base = Path(get_curr_dir(__file__)).joinpath("manifests")

    ##Variables
    LANG = args.lang.lower()
    assert LANG == "eng" or LANG == "ita"
    TRAIN_MANIFEST = str(path_base.joinpath(f"train_{LANG}_manifest.json"))

    ##Creating config file
    config_path = Path(get_curr_dir(__file__)).joinpath("manifests", args.config)
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=False)
    config = OmegaConf.create(config)
    LABELS = list(map(lambda x: int(x), command_eng.keys()))
    LABELS.append(int(len(command_eng)))
    batch_size = config.model.train_ds.batch_size
    config.model.labels = LABELS
    config.model.lang = LANG

    config.model.train_ds.augmentor.mynoise.dataset_path_real = str(Path(get_curr_dir(__file__)).joinpath("dataset/Dataset_real_conv"))
    if args.synth:
        config.model.train_ds.augmentor.mynoise.dataset_path_synth = str(Path(get_curr_dir(__file__)).joinpath("dataset/Dataset_synth_conv"))
    else:
        config.model.train_ds.augmentor.mynoise.dataset_path_synth = None
    #config.model.train_ds.augmentor.mynoise.dataset_path_reject = str(Path(get_curr_dir(__file__)).joinpath("dataset/common_voices"))

    config.model.train_ds.manifest_filepath = TRAIN_MANIFEST

    OmegaConf.resolve(config)
    sample_rate = config.model.sample_rate
    print("Trainer config - \n")
    print(OmegaConf.to_yaml(config.trainer))

    cuda = 1 if torch.cuda.is_available() else 0
    config.trainer.gpus = cuda
    config.trainer.accelerator = None

    validation_callback = ValidationCallback(config.model.train_ds.augmentor.mynoise, config.model.labels, batch_size, args.train_id)
    CALLBACKS = [validation_callback]
    # CALLBACKS = []
    trainer = pl.Trainer(**config.trainer, callbacks=CALLBACKS)
    asr_model = Model(cfg=config.model, trainer=trainer, class_weight=None) #Modify here to set class weight

    exp_dir = exp_manager(trainer, config.get("exp_manager", None))
    exp_dir = str(exp_dir)
    if args.logfile:
        logger.exp_dir = exp_dir
    print(exp_dir)
    validation_callback.exp_dir = exp_dir

    if platform.system().lower() == "windows":
        logdir = Path(exp_dir).relative_to(get_curr_dir(__file__))
        cmd = ["tensorboard", "--logdir", logdir]
        p = subprocess.Popen(cmd)

    ##Training
    print(f"Starting training for {LANG}\n")
    trainer.fit(asr_model)

    logger.close()


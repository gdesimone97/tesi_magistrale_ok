import json
import shutil
import os
import numpy as np
import torch
import platform
from pathlib import Path
from pytorch_lightning.callbacks.base import Callback
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import Sampler
from nemo.collections.asr.modules import ConvASRDecoderClassification
from nemo.core.classes.common import typecheck
from nemo.collections.asr.models.classification_models import EncDecClassificationModel
from nemo.collections.asr.parts.preprocessing.perturb import Perturbation, register_perturbation, process_augmentations
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.data import audio_to_label_dataset
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.common.losses import CrossEntropyLoss
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import copy
from preprocessing import Preprocessing
from global_utils import get_curr_dir
from utils import SingletonMeta, is_windows, extract_logits
import pandas as pd
import random
import soundfile as sf
from typing import Any, Callable, Dict, Optional, Union
from pytorch_lightning.core.saving import inspect, parse_class_init_keys, CHECKPOINT_PAST_HPARAMS_KEYS, \
    _convert_loaded_hparams, load_hparams_from_tags_csv, load_hparams_from_yaml, pl_load
from pytorch_lightning.utilities import rank_zero_warn
from omegaconf import OmegaConf
import logging

MAXSIZE = 2**32-1
'''
This class implements the PyTorch callback interface.
It have to use in both train phase and test phase.
Before fit starts, it copy the audio clips, train, validation and test set files into experiment dir.
When epoch end it performs the validation step, analyzing the model performance at different snr levels 
'''
class ValidationCallback(Callback):
    '''
    Builds the class for validation phase.
    In this function the Augmentation object is created
    :param cfg configuration of mynoise augmentator contained into yaml configuration file like dict object
    :param labels list of labels
    :param batch_size
    :param id of the training
    '''
    def init_validation(self, cfg, labels, batch_size, train_id):
        cfg = copy.deepcopy(cfg)
        del cfg["prob"]
        lang = cfg["lang"]
        self.lang = lang
        self.batch_size = batch_size
        cfg["batch_size"] = self.batch_size
        cfg["train_id"] = train_id
        self.test = False
        self._exp_dir = None
        working_dir = Path(get_curr_dir(__file__)).joinpath(f".working_dir/train_id_{train_id}")
        self.working_dir = working_dir
        self.augmentation = Augmentation(**cfg)
        self.validation_path = Path(get_curr_dir(__file__)).joinpath(f"manifests/validation_{self.lang}_manifest.json")
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.validation_step = 0
        self.num_cpu = 6
        self.validation_df = self.augmentation.preprocessing.validation_set
        self.snr_range = range(self.augmentation.min_snr_val,
                               self.augmentation.max_snr_val + self.augmentation.snr_val_step,
                               self.augmentation.snr_val_step)
        self.validation_dataset = ValidationDataset(self.augmentation.sample_rate, labels)
        self._dataloaders = self.get_dataloaders()

    '''
    Builds the class for test phase
    :param labels list of labels
    :param batch_size
    :param snr_range range object containing the snr values to test
    :param sample_rate
    :param lang language, must be eng or ita
    :param seed
    :param max_gain_db 
    :param noise_test_path path to noise test set, it must be a csv
    :param test_manifest_path path to test set file, it must ba a csv
    '''
    def init_test(self, labels, batch_size, snr_range, sample_rate, lang, seed, max_gain_db, noise_test_path, test_manifest_path):
        self.batch_size = batch_size
        working_dir = Path(get_curr_dir(__file__)).joinpath(".working_dir/test")
        self.augmentation = Augmentation(seed=seed, sample_rate=sample_rate, max_gain_db=max_gain_db, lang=lang,
                                         train_frac=None, val_frac=None, test_frac=None,
                                         min_snr=None, max_snr=None, dataset_path_real=None,
                                         test=True, batch_size=self.batch_size)
        self.working_dir = working_dir
        self.lang = lang
        self.test = True
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.validation_path = str(test_manifest_path).split(".")[0] + ".json"
        self.validation_df = pd.read_csv(test_manifest_path, index_col=0)
        self.validation_df = self.check_path_validation(self.validation_df)
        noise_set = pd.read_csv(noise_test_path, index_col=0)
        noise_set = self.check_path_validation(noise_set, noise_set=True)
        self.augmentation.noise_df_test = noise_set
        self.num_cpu = 6
        self.snr_range = snr_range
        self.validation_dataset = ValidationDataset(sample_rate, labels)
        self._dataloaders = self.get_dataloaders()

    '''
    Get the class weight for the training set
    '''
    @property
    def class_weight_train(self):
        return self.augmentation.preprocessing.class_weight_train
    '''
    Gets the dataloaders
    '''
    @property
    def dataloaders(self):
        return self._dataloaders
    '''
    Gets the set experiment directory
    '''
    @property
    def exp_dir(self):
        if self._exp_dir is None:
            raise Exception("You need to set the exp_dir param before to start the fit")
        return self._exp_dir
    '''
    Sets the experiment directory 
    '''
    @exp_dir.setter
    def exp_dir(self, exp_dir):
        self._exp_dir = exp_dir

    '''
    Builds the class respect to tran or test phase.
    To configure the class for the test phase set to True the key argument "test"
    '''
    def __init__(self, *args, **kwargs):
        test = kwargs.get("test", False)
        if test:
            labels, batch_size, snr_range, sample_rate, lang, seed, max_gain_db, noise_test_path, test_manifest_path = args
            self.init_test(labels, batch_size, snr_range, sample_rate, lang, seed, max_gain_db, noise_test_path, test_manifest_path)
        else:
            cfg, labels, batch_size, train_id = args
            self.init_validation(cfg, labels, batch_size, train_id)
    '''
    Converts the audio file path contained into csv file from Linux to Windows platform
    :param validation_df csv file that contains the paths to convert
    :param noise_set if the csv file corresponds to noise set, it must be True else False
    '''
    def check_path_validation(self, validation_df: pd.DataFrame, noise_set=False):
        split_param = "fsd" if noise_set else "dataset"
        validation_df_out = pd.DataFrame(columns=validation_df.columns)
        for index, row in validation_df.iterrows():
            audio_path = Path(row["path"])
            if audio_path.anchor != r"C:\\":
                path = str(audio_path).split(split_param)[1]
                if path[0] == '\\':
                    path = path[1:]
                nw_path = Path(get_curr_dir(__file__)).joinpath(split_param, path)
                if "ssd_link" in nw_path.parts:
                    nw_path = Path(str(nw_path).replace("ssd_link", ""))
                validation_df_out = validation_df_out.append(validation_df.iloc[index])
                validation_df_out.at[index, "path"] = nw_path
        return validation_df_out

    '''
    Adds the noise to samples at different snr levels
    '''
    def create_augmentation_manifest(self):
        working_dir = self.working_dir
        clips_dir = Path(working_dir).joinpath("clips")
        if working_dir.exists():
            if working_dir.is_symlink():
                path = os.path.relpath(working_dir)
                for root, dirs, files in os.walk(path):
                    for file in files:
                        os.remove(Path(root).joinpath(file))
                    for dir in dirs:
                        shutil.rmtree(Path(root).joinpath(dir))
            else:
                shutil.rmtree(working_dir)
        working_dir.mkdir(exist_ok=True, parents=True)
        clips_dir.mkdir(exist_ok=True, parents=True)
        path_db = {}
        for snr in self.snr_range:
            path_db[snr] = ([], [])
        for index, row in self.validation_df.iterrows():
            audio_path_input = row["path"]
            cmd_index = row["cmd_index"]
            speaker = row["speaker_id"]
            service = row["service"]
            audio_name = Path(audio_path_input).name
            if service == "real":
                audio_path_input = Path(get_curr_dir(__file__)).joinpath(f"dataset/Dataset_real_conv/{speaker}/{self.lang}/{audio_name}")
            elif pd.isna(service):
                audio_path_input = Path(get_curr_dir(__file__)).joinpath(f"dataset/common_voices/{self.lang}/clips/{audio_name}")
            else:
                raise Exception(f"service not found. Received: {service}")
            audio = AudioSegment.from_file(audio_path_input, target_sr=self.augmentation.sample_rate)
            for snr in self.snr_range:
                snr_dir = Path(clips_dir).joinpath(str(snr))
                snr_dir.mkdir(parents=True, exist_ok=True)
                audio_perturb = self.augmentation.perturb_validation(audio, snr)
                audio_path_out = Path(clips_dir).joinpath(snr_dir, (str(speaker)+"_"+audio_name))
                if audio_path_out.exists():
                    raise Exception("Audio already exists")
                sf.write(audio_path_out, audio_perturb.samples, samplerate=self.augmentation.sample_rate, format="WAV")
                path_db[snr][0].append(audio_path_out)
                path_db[snr][1].append(cmd_index)
        for k, v in path_db.items():
            manifest_path = working_dir.joinpath(f"validation_{k}.json") if not self.test else working_dir.joinpath(f"test_{k}.json")
            fil = open(manifest_path, "w")
            path_list = v[0]
            command_list = v[1]
            assert len(path_list) == len(command_list)
            for i in range(len(path_list)):
                data = {"audio_filepath": str(path_list[i]), "duration": 0, "command": command_list[i]}
                json.dump(data, fil)
                fil.write("\n")
            fil.close()

    '''
    Creates a dataloader for each snr level
    '''
    def get_dataloaders(self) -> dict:
        dataloaders = {}
        self.create_augmentation_manifest()
        for root, dirs, files in os.walk(self.working_dir):
            for fil in files:
                snr = fil.split("_")[1].split(".")[0]
                manifest_path = Path(root).joinpath(fil)
                dataloader = self.validation_dataset.get_dataloader(manifest_path, batch_size=self.batch_size)
                dataloaders[snr] = dataloader
            break
        return dataloaders
    '''
    Computes the validation loss, logtis and predicted labels
    :param model model to validate
    :param dataloader
    :param device to use cpu or gpu
    '''
    def validate(self, model, dataloader, device="gpu"):
        try:
            logits, true_labels = extract_logits(model, dataloader, device=device)
        except RuntimeError:
            logits, true_labels = extract_logits(model, dataloader, device="cpu")
        predict_labels = model.predict(logits)
        logits = logits.cpu()
        true_labels = true_labels.cpu()
        predict_labels = predict_labels.cpu()
        loss = self.cross_entropy_loss(logits.float(), true_labels).item()
        true_labels = true_labels.cpu().detach().numpy()
        predict_labels = predict_labels.cpu().detach().numpy()
        predict_labels = np.argmax(predict_labels, axis=1)
        return loss, true_labels, predict_labels, logits

    '''
    Before fit starts, it copy the audio clips, train, validation and test set files into experiment dir.
    '''
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        exp_dir = Path(self._exp_dir)
        exp_dir_db = exp_dir.joinpath("db")
        exp_dir_clips = exp_dir_db.joinpath("clips")
        exp_dir_clips.mkdir(parents=True)
        root, dirs, files = next(iter(os.walk(self.working_dir)))
        for file in files:
            if file.split(".")[1] != "json": continue
            shutil.copy(Path(root).joinpath(file), exp_dir_db.joinpath(file))
        if int(platform.python_version_tuple()[1]) == 8:
            shutil.copytree(Path(self.working_dir).joinpath("clips"), exp_dir_clips, dirs_exist_ok=True)
        else:
            for dir in dirs:
                dirname = dir
                dir = Path(root).joinpath(dir)
                shutil.copytree(dir, exp_dir_clips.joinpath(dirname))
        manifests_path = Path(get_curr_dir(__file__)).joinpath("manifests")
        manifests_list = ["train", "validation", "test"]
        for e in manifests_list:
            path = manifests_path.joinpath(f"{e}_{self.lang}_manifest")
            path_json = str(path) + ".json"
            path_csv = str(path) + ".csv"
            shutil.copy(path_json, exp_dir_db)
            shutil.copy(path_csv, exp_dir_db)
        for e in manifests_list:
            path = manifests_path.joinpath(f"noise_{e}_{self.lang}_manifest.csv")
            shutil.copy(path, exp_dir_db)

    '''
    When epoch end it performs the validation step, analyzing the model performance at different snr levels.
    It computes the accuracy and the balanced accuracy at different snr levels and the mean validation loss
    '''
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused = None) -> None:
        model: Model = trainer.model
        if is_windows():
            model = model.cpu()
        model.training = False
        stats = {}
        for snr, dataloader in self.dataloaders.items():
            res = self.validate(model, dataloader=dataloader, device="gpu")
            true_labels = res[1]
            predict_labels = res[2]
            bal_acc = balanced_accuracy_score(y_true=true_labels, y_pred=predict_labels)
            acc = accuracy_score(y_true=true_labels, y_pred=predict_labels)
            stats[snr] = (res[0], bal_acc, acc)
        acc = list(map(lambda x: x[2], stats.values()))
        bal_acc = list(map(lambda x: x[1], stats.values()))
        loss = list(map(lambda x: x[0], stats.values()))
        acc = np.array(acc)
        bal_acc = np.array(bal_acc)
        loss = np.array(loss)
        accuracy_mean = acc.mean()
        bal_acc_mean = bal_acc.mean()
        loss_mean = loss.mean()
        trainer.logger.log_metrics({f"accuracy_mean": accuracy_mean}, step=self.validation_step)
        trainer.logger.log_metrics({f"bal_acc_mean": bal_acc_mean}, step=self.validation_step)
        pl_module.log("val_loss", loss_mean)
        for k, v in stats.items():
            acc = v[2]
            bal_acc = v[1]
            trainer.logger.log_metrics({f"accuracy_snr_{k}dB": acc}, step=self.validation_step)
            trainer.logger.log_metrics({f"bal_acc_snr_{k}dB": bal_acc}, step=self.validation_step)
        model.save_backup(self.exp_dir, epoch=self.validation_step, loss=loss_mean)
        self.validation_step += 1
        model = model.cuda()
        model.training = True

'''
Overrides the NeMo Decoder 
'''
class Decoder(ConvASRDecoderClassification):

    @typecheck()
    def forward(self, encoder_output):
        batch, in_channels, timesteps = encoder_output.size()

        encoder_output = self.pooling(encoder_output).view(batch, in_channels)  # [B, C]
        logits = self.decoder_layers(encoder_output)  # [B, num_classes]

        if not self._return_logits:
            raise Exception("return logits param must be TRUE")

        return logits

'''
Defines a custom EncDecClassificationModel
'''
class Model(EncDecClassificationModel):
    '''
    Builds the model
    :param cfg dictonary containing the configuration of the model, see the yaml file for more details
    :param trainer Pytorch trainer
    :param class_weight class weights to set into loss function
    :param loading True if the user is loading a model, False if he is creating the model
    '''
    def __init__(self, cfg, trainer = None, class_weight=None, loading=False):
        try:
            self.last_layer = cfg.last_layer
        except Exception:
            self.last_layer = None
        self.class_weight = class_weight
        if self.last_layer == None:
            print("last_layer param not set, will be set the softmax")
        elif self.last_layer != "sigmoid" and self.last_layer != "softmax":
            raise Exception("Last layer must be softmax or sigmoid")
        if "last_layer" in cfg.keys():
            del cfg["last_layer"]
        if not loading:
            self._load_dataset_csv(cfg)
        if loading:
            cfg.train_ds = None
        super().__init__(cfg, trainer)

    '''
    Loads the train set
    '''
    def _load_dataset_csv(self, cfg):
        mainfest_path = Path(cfg.train_ds.manifest_filepath)
        csv_path = mainfest_path.parent.joinpath(f"train_{cfg.lang}_manifest.csv")
        self.dataset_csv = pd.read_csv(csv_path, index_col=0)

    '''
    Predicts the labels. Takes the logits as input
    '''
    def predict(self, logits):
        if self.last_layer == "softmax" or self.last_layer is None:
            predicted_values = torch.nn.functional.softmax(logits, dim=-1)
        else:
            predicted_values = torch.nn.functional.sigmoid(logits)
        return predicted_values

    def validation_step(self, batch, batch_idx, dataloader_idx=0): #Get singleton class
        raise Exception("Use the validation callback to perform the validation task")

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        raise Exception("Use the validation callback to perform the validation task")

    '''
    Sets the loss function
    '''
    def _setup_loss(self):
        weights = list(self.class_weight) if self.class_weight is not None else None
        return CrossEntropyLoss(weight=weights)

    '''
    Serializes the model into a file
    '''
    def save_backup(self, exp_dir, epoch ,loss=None):
        save_dict = {}
        save_dict["state_dict"] = self.state_dict()
        save_dict["cfg"] = self.cfg
        text = "matchcboxnet--val_loss={}-epoch={}.model"
        text = text.format(round(loss, 4), epoch) if loss is not None else text.format("nope", epoch)
        path = Path(exp_dir).joinpath("checkpoints_backup")
        path.mkdir(exist_ok=True, parents=True)
        path_save = path.joinpath(text)
        torch.save(save_dict, path_save)

    '''
    Loads the model by a file
    '''
    @classmethod
    def load_backup(cls, ckpt_name, exp_dir):
        ckpt_path = Path(exp_dir).joinpath("checkpoints_backup", ckpt_name)
        checkpoint = torch.load(ckpt_path)
        state_dict = checkpoint["state_dict"]
        class_weight = state_dict["loss.weight"] if "loss.weight" in state_dict.keys() else None
        # del state_dict["loss.weight"]
        model = cls(checkpoint["cfg"], class_weight=class_weight, loading=True)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return model

    '''
    Creates a model object using the information contained into the checkpoint 
    (No calls it directly, use the load function provided by the Nemo framework. Loads only nemo files) 
    '''
    @classmethod
    def _load_model_state(cls, checkpoint: Dict[str, Any], strict: bool = True, **cls_kwargs_new):
        cls_spec = inspect.getfullargspec(cls.__init__)
        cls_init_args_name = inspect.signature(cls.__init__).parameters.keys()

        self_var, args_var, kwargs_var = parse_class_init_keys(cls)
        drop_names = [n for n in (self_var, args_var, kwargs_var) if n]
        cls_init_args_name = list(filter(lambda n: n not in drop_names, cls_init_args_name))

        cls_kwargs_loaded = {}
        # pass in the values we saved automatically
        if cls.CHECKPOINT_HYPER_PARAMS_KEY in checkpoint:

            # 1. (backward compatibility) Try to restore model hparams from checkpoint using old/past keys
            for _old_hparam_key in CHECKPOINT_PAST_HPARAMS_KEYS:
                cls_kwargs_loaded.update(checkpoint.get(_old_hparam_key, {}))

            # 2. Try to restore model hparams from checkpoint using the new key
            _new_hparam_key = cls.CHECKPOINT_HYPER_PARAMS_KEY
            cls_kwargs_loaded.update(checkpoint.get(_new_hparam_key))

            # 3. Ensure that `cls_kwargs_old` has the right type, back compatibility between dict and Namespace
            cls_kwargs_loaded = _convert_loaded_hparams(
                cls_kwargs_loaded, checkpoint.get(cls.CHECKPOINT_HYPER_PARAMS_TYPE)
            )

            # 4. Update cls_kwargs_new with cls_kwargs_old, such that new has higher priority
            args_name = checkpoint.get(cls.CHECKPOINT_HYPER_PARAMS_NAME)
            if args_name and args_name in cls_init_args_name:
                cls_kwargs_loaded = {args_name: cls_kwargs_loaded}

        _cls_kwargs = {}
        _cls_kwargs.update(cls_kwargs_loaded)
        _cls_kwargs.update(cls_kwargs_new)

        if not cls_spec.varkw:
            # filter kwargs according to class init unless it allows any argument via kwargs
            _cls_kwargs = {k: v for k, v in _cls_kwargs.items() if k in cls_init_args_name}

        _cls_kwargs["cfg"]["train_ds"]["augmentor"] = None
        loss_weight = checkpoint["state_dict"].get("loss.weight")
        loss_weight = loss_weight.cpu().detach().numpy()
        _cls_kwargs["class_weight"] = loss_weight
        _cls_kwargs["loading"] = True
        model = cls(**_cls_kwargs)

        # give model a chance to load something
        model.on_load_checkpoint(checkpoint)

        # load the state_dict on the model automatically
        keys = model.load_state_dict(checkpoint["state_dict"], strict=strict)

        if not strict:
            if keys.missing_keys:
                rank_zero_warn(
                    f"Found keys that are in the model state dict but not in the checkpoint: {keys.missing_keys}"
                )
            if keys.unexpected_keys:
                rank_zero_warn(
                    f"Found keys that are not in the model state dict but in the checkpoint: {keys.unexpected_keys}"
                )

        return model

    '''
    Like the original function but with the following changes:
    1. Created a own featurizer
    2. Sets a own sampler into the dataloader creation 
    '''
    def _setup_dataloader_from_config(self, config):

        OmegaConf.set_struct(config, False)
        config.is_regression_task = self.is_regression_task
        OmegaConf.set_struct(config, True)

        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        featurizer = MyWaveFormFeaturizer(
            sample_rate=config['sample_rate'], int_values=config.get('int_values', False), augmentor=augmentor
        )
        shuffle = config['shuffle']

        # Instantiate tarred dataset loader or normal dataset loader
        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                logging.warning(
                    "Could not load dataset as `manifest_filepath` is None or "
                    f"`tarred_audio_filepaths` is None. Provided config : {config}"
                )
                return None

            if 'vad_stream' in config and config['vad_stream']:
                logging.warning("VAD inference does not support tarred dataset now")
                return None

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
            dataset = audio_to_label_dataset.get_tarred_classification_label_dataset(
                featurizer=featurizer,
                config=OmegaConf.to_container(config),
                shuffle_n=shuffle_n,
                global_rank=self.global_rank,
                world_size=self.world_size,
            )
            shuffle = False
            batch_size = config['batch_size']
            collate_func = dataset.collate_fn

        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` is None. Provided config : {config}")
                return None

            if 'vad_stream' in config and config['vad_stream']:
                logging.info("Perform streaming frame-level VAD")
                dataset = audio_to_label_dataset.get_speech_label_dataset(
                    featurizer=featurizer, config=OmegaConf.to_container(config)
                )
                batch_size = 1
                collate_func = dataset.vad_frame_seq_collate_fn
            else:
                dataset = audio_to_label_dataset.get_classification_label_dataset(
                    featurizer=featurizer, config=OmegaConf.to_container(config)
                )
                batch_size = config['batch_size']
                collate_func = dataset.collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=DatasetSampler(dataset, batch_size, self.cfg.labels, self.cfg.train_ds.augmentor.mynoise.seed),
            collate_fn=collate_func,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    '''
    Loads a NeMo checkpoint
    '''
    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        *args,
        map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
        hparams_file: Optional[str] = None,
        strict: bool = True,
        **kwargs,
    ):
        try:
            register_perturbation("mynoise", Augmentation)
        except KeyError: pass

        if map_location is not None:
            checkpoint = pl_load(checkpoint_path, map_location=map_location)
        else:
            checkpoint = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)

        if hparams_file is not None:
            extension = hparams_file.split(".")[-1]
            if extension.lower() == "csv":
                hparams = load_hparams_from_tags_csv(hparams_file)
            elif extension.lower() in ("yml", "yaml"):
                hparams = load_hparams_from_yaml(hparams_file)
            else:
                raise ValueError(".csv, .yml or .yaml is required for `hparams_file`")

            hparams["on_gpu"] = False

            # overwrite hparams by the given file
            checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = hparams

        # for past checkpoint need to add the new key
        if cls.CHECKPOINT_HYPER_PARAMS_KEY not in checkpoint:
            checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = {}
        # override the hparams with values that were passed in
        structure = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]
        for k, v in kwargs.items(): #Adding last_layer
            OmegaConf.update(structure, f"{k}", v, force_add=True)

        model = cls._load_model_state(checkpoint, strict=strict, **kwargs)
        return model


'''
This class allows to user to add noise perturbation to audio command.
It can be used like augmentor into Nvidia Nemo framework
'''
class Augmentation(Perturbation, metaclass=SingletonMeta):
    '''
    :param seed: seed for the random tasks
    :param sample_rate: sampling rate used for the net training
    :param max_gain_db: maximum gain can be add to audio command during the perturbation step
    :param train_frac: fraction of dataset used to compose the train set
    :param val_frac: fraction of dataset used to compose the validation set
    :param test_frac: fraction of dataset used to compose the test set
    :param min_snr: minimum snr level to use for the audio command perturbation, this param is used only during
                    the training task
    :param max_snr: maximum snr level to use for the audio command perturbation, this param is used only during
                    the training task
    :param dataset_path_real: path to real audio command files
    :param dataset_path_synth: path to synthetic audio command files, if None this files will be ignored
    :param dataset_path_reject: path to files to perform the "rejection" task
    :param lang: commands language, can be "ita" or "eng"
    '''
    def __init__(self, seed, sample_rate, max_gain_db, train_frac, val_frac, test_frac, min_snr, max_snr,
                 dataset_path_real,  lang, batch_size, train_id=None,
                 dataset_path_reject=None, dataset_path_synth=None, test=False):
        if not test:
            self.preprocessing = Preprocessing(train_frac=train_frac, val_frac=val_frac, test_frac=test_frac,
                                               dataset_path_real=dataset_path_real,
                                               dataset_path_synth= dataset_path_synth,
                                               dataset_path_reject=dataset_path_reject,
                                               sample_rate=sample_rate, seed=seed, lang=lang, batch_size=batch_size,
                                               train_id=train_id)
        self.test = test
        self.seed = seed
        self.noise_instances = {}
        self.noise_instances_validation = {}
        self.rng = random.Random(self.seed)
        self._sample_rate = sample_rate
        self.min_snr, self.max_snr = max_snr,  min_snr
        self.min_snr_val, self.max_snr_val, self.snr_val_step = 0, 30, 5
        self._max_gain_db = max_gain_db
        if not test:
            self._noise_df_train = self.preprocessing.noise_train_set
            self._noise_df_val = self.preprocessing.noise_validation_set
            self._noise_df_test = self.preprocessing.noise_test_set

    @property
    def noise_df_train(self):
        return self._noise_df_train

    @property
    def noise_df_val(self):
        return self._noise_df_val

    @property
    def noise_df_test(self):
        return self._noise_df_test

    @noise_df_train.setter
    def noise_df_train(self, path):
        if not self.test:
            raise Exception("Cannot modify this path during the train step")
        self._noise_df_train = path

    @noise_df_val.setter
    def noise_df_val(self, path):
        if not self.test:
            raise Exception("Cannot modify this path during the train step")
        self._noise_df_val = path

    @noise_df_test.setter
    def noise_df_test(self, path):
        if not self.test:
            raise Exception("Cannot modify this path during the train step")
        self._noise_df_test = path

    @property
    def sample_rate(self):
        return self._sample_rate

    '''
    Samples a noise sample based on the specified dataset split type.
    The data set subdivision type can be one of: "train", "val", "test"
    '''
    def sample(self, dataset_split: str, rng=None):
        rng = self.rng if rng is None else rng
        if dataset_split == "train": df = self.noise_df_train
        elif dataset_split == "val": df = self.noise_df_val
        elif dataset_split == "test": df = self.noise_df_test
        else: raise Exception("Wrong dataset specified")
        weights = self._weight_balancing(df)
        data = df.sample(n=1, weights=weights, random_state=rng.randint(0, MAXSIZE))
        return data["path"].tolist()[0]
    '''
    Gets a noise sampling it by the noise validation set
    :param hash of command sample
    '''
    def get_validation_noise(self, data_hash):
        return self.sample(dataset_split="val", rng=random.Random(data_hash))

    '''
    Gets a noise sampling it by the noise test set
    '''
    def get_test_noise(self):
        return self.sample(dataset_split="test", rng=self.rng)

    '''
    Computes the weights for the dataset balancing
    '''
    def _weight_balancing(self, df: pd.DataFrame):
        return compute_sample_weight("balanced", y=df["noise_type"].to_numpy())

    '''
    Perturbs the audio data with the given noise at the specified snr
    '''
    @staticmethod
    def perturb_test(data, snr, noise, seed, max_gain_db=300) -> AudioSegment:
        def perturb_with_input_noise(data, noise, snr_db, start_time, rng, data_rms=None):
            if data_rms is None:
                data_rms = data.rms_db
            noise_gain_db = min(data_rms - noise.rms_db - snr_db, max_gain_db)

            # calculate noise segment to use
            if noise.duration > (start_time + data.duration):
                noise.subsegment(start_time=start_time, end_time=start_time + data.duration)

            # adjust gain for snr purposes and superimpose
            noise.gain_db(noise_gain_db)

            if noise._samples.shape[0] < data._samples.shape[0]:
                noise_idx = rng.randint(0, data._samples.shape[0] - noise._samples.shape[0])
                data._samples[noise_idx: noise_idx + noise._samples.shape[0]] += noise._samples
            else:
                data._samples += noise._samples

        diff = noise.duration - data.duration
        start_time = random.Random(seed).uniform(0, diff) if diff > 0 else 0
        if start_time >= data.duration:
            start_time = 0
        data_copy = copy.deepcopy(data)
        perturb_with_input_noise(data_copy, copy.deepcopy(noise), snr, start_time, rng=random.Random(seed))
        return data_copy

    '''
    Samples a noise sample and performs the audio perturbation supplied as input at different levels of snr.
    The starting point for the noise subsegment is chosen randomly
    '''
    def perturb_validation(self, data: AudioSegment, snr, noise=None):
        data_hash = hash(tuple(data.samples.tolist()))
        if noise is None:
            if self.test:
                noise_path = self.get_test_noise()
                suffx = str(noise_path).split("noise_set")[1]
                suffx = suffx[1:]
                noise_path = Path(get_curr_dir(__file__)).joinpath("fsd", "noise_set", suffx)
            else:
                noise_path = self.get_validation_noise(data_hash)
            instance = self.noise_instances_validation.get(noise_path, None)
            if instance is None:
                noise = AudioSegment.from_file(noise_path, target_sr=data.sample_rate)
                self.noise_instances_validation[noise_path] = noise
            else:
                noise = instance

        diff = noise.duration - data.duration
        start_time = random.Random(self.seed).uniform(0, diff) if diff > 0 else 0
        if start_time >= data.duration:
            start_time = 0
        data_copy = copy.deepcopy(data)
        self.perturb_with_input_noise(data_copy, copy.deepcopy(noise), snr, start_time)
        return data_copy

    '''
    Samples a noise sample and performs the audio perturbation supplied as input.
    The snr level is chosen randomly
    The starting point for the noise subsegment is chosen randomly
    '''
    def perturb(self, data: AudioSegment, dataset_split="train"):
        assert self.sample_rate == data.sample_rate
        noise_path = self.sample(dataset_split)
        instance = self.noise_instances.get(noise_path, None)
        if instance is None:
            noise = AudioSegment.from_file(noise_path, target_sr=data.sample_rate)
            self.noise_instances[noise_path] = noise
        else:
            noise = instance
        diff = noise.duration - data.duration
        if diff <= 0:
            start_time = 0
        else:
            start_time = self.rng.uniform(0.0, diff)
        snr = self.rng.uniform(self.min_snr, self.max_snr)
        self.perturb_with_input_noise(data, copy.deepcopy(noise), snr, start_time)

    '''
    Performs audio perturbation with the snr-level noise sample supplied as an input
    '''
    def perturb_with_input_noise(self, data, noise, snr_db, start_time, data_rms=None):
        rng = self.rng
        if data_rms is None:
            data_rms = data.rms_db
        noise_gain_db = min(data_rms - noise.rms_db - snr_db, self._max_gain_db)

        # calculate noise segment to use
        if noise.duration > (start_time + data.duration):
            noise.subsegment(start_time=start_time, end_time=start_time + data.duration)

        # adjust gain for snr purposes and superimpose
        noise.gain_db(noise_gain_db)

        if noise._samples.shape[0] < data._samples.shape[0]:
            noise_idx = rng.randint(0, data._samples.shape[0] - noise._samples.shape[0])
            data._samples[noise_idx: noise_idx + noise._samples.shape[0]] += noise._samples
        else:
            data._samples += noise._samples

'''
This class creates a dataloader by the given manifest path (only json files)
'''
class ValidationDataset:
    def __init__(self, sample_rate, labels):
        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=False, augmentor=None)
        self.labels = labels

    def get_dataloader(self, manifest_path, batch_size):
        manifest_path = str(manifest_path)
        config = {
            "manifest_filepath": manifest_path,
            "labels": self.labels,
            "max_duration": None,
            "min_duration": None,
            "trim": False,
            "is_regression_task": False
        }
        dataset = audio_to_label_dataset.get_classification_label_dataset(self.featurizer, config)
        collate_func = dataset.collate_fn
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_func,
            drop_last=False,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        return dataloader
'''
Like the inherit class but this class saves the loaded audios in memory to accelerates the model training
'''
class MyWaveFormFeaturizer(WaveformFeaturizer):
    def __init__(self, sample_rate=16000, int_values=False, augmentor=None):
        self.instances = {}
        super(MyWaveFormFeaturizer, self).__init__(sample_rate, int_values, augmentor)
    def process(self, file_path, offset=0, duration=0, trim=False, orig_sr=None):
        instance = self.instances.get(file_path, None)
        if instance is None:
            audio = AudioSegment.from_file(
                file_path,
                target_sr=self.sample_rate,
                int_values=self.int_values,
                offset=offset,
                duration=duration,
                trim=trim,
                orig_sr=orig_sr,
            )
            self.instances[file_path] = audio
        else:
            audio = copy.deepcopy(self.instances[file_path])
        return self.process_segment(audio)

'''
Implements the Pythorch Sampler interface
Performs the batch balancing
'''
class DatasetSampler(Sampler):
    def __init__(self, audio_to_classification_dataset, batch_size, labels, seed):
        audio_to_classification_dataset = audio_to_classification_dataset.collection
        self.dataset = pd.DataFrame(columns=["cmd_index", "index"])
        data = [(x.label, i) for i, x in enumerate(audio_to_classification_dataset)]
        self.dataset = self.dataset.append(pd.DataFrame(data, columns=self.dataset.columns))
        self.rng = np.random.RandomState(seed)
        self.batch_size = batch_size
        self.balance_dataset = BalanceDataset(self.dataset, self.batch_size, seed)

    def __len__(self):
        return len(self.dataset)
    def __iter__(self):
        for index in self.get_batch():
            yield index
    def balance(self):
        df_out = self.balance_dataset.balancing()
        return df_out
    def get_batch(self):
        df = self.balance()
        return df["index"].tolist()

'''
This class creates a copy of dataset. 
In this copy each batch contains the same number of samples per class.
Each class has its own iterator.
At each iteration the iterators of the classes are chosen sequentially. 
When a class iterator ends, it is reset
'''
class BalanceDataset:
    def __init__(self, dataset: pd.DataFrame, batch_size, seed):
        self.train_set = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.batch_db = self._split()
        self.batch_iter = self._iterate()

    def _split(self):
        cmds = list(self.train_set.groupby("cmd_index").groups.keys())
        cmds.sort()
        batch_db = {}
        for k in cmds:
            batch_db[k] = self.train_set.loc[self.train_set["cmd_index"] == k]
        return batch_db
    def _iterate(self):
        batch_iter = {}
        for k, v in self.batch_db.items():
            batch_iter[k] = v.iterrows()
        return batch_iter
    def log(self, batch):
        for k, v in batch.groupby("cmd_index").groups.items():
            print(f"{k}:\t{len(v)}", end="\t")
        print()
        print("*" * 50)
    def balancing(self):
        rng = np.random.RandomState(self.seed)
        batch = pd.DataFrame(columns=self.train_set.columns)
        db_out = pd.DataFrame(columns=self.train_set.columns)
        for j in range(len(self.train_set)):
            k = j % len(self.batch_db.keys())
            batch_iterator = self.batch_iter[k]
            try:
                sample = next(batch_iterator)[1]
            except StopIteration:
                self.batch_iter[k] = iter(self.batch_db[k].iterrows())
                batch_iterator = self.batch_iter[k]
                sample = next(batch_iterator)[1]
            batch = batch.append(sample)
            if len(batch) >= self.batch_size:
                batch = batch.sample(frac=1, random_state=rng)
                # self.log(batch)
                db_out = db_out.append(batch, ignore_index=True)
                batch = pd.DataFrame(columns=self.train_set.columns)
        return db_out
import numpy as np
import shutil
import soundfile
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
import commands
from pathlib import Path
from global_utils import get_curr_dir, select_analyzator, EXCLUDES
import pandas as pd
import os
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
import soundfile as sd
import random
import librosa
import json
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from torch.utils.data import BatchSampler, WeightedRandomSampler
import torch
import gc

'''
This class splits the dataset into train set, validation set, and test set.
The same task if performed for the noise set
'''
class Preprocessing:
    '''
    :param seed: seed for the random tasks
    :param sample_rate: sampling rate used for the net training
    :param train_frac: fraction of dataset used to compose the train set
    :param val_frac: fraction of dataset used to compose the validation set
    :param test_frac: fraction of dataset used to compose the test set
    :param dataset_path_real: path to real audio command files
    :param dataset_path_synth: path to synthetic audio command files, if None this files will be ignored
    :param dataset_path_reject: path to files to perform the "rejection" task
    :param lang: commands language, can be "ita" or "eng"
    '''
    def __init__(self, seed, sample_rate, train_frac, val_frac, test_frac,
                 dataset_path_real, dataset_path_reject: str,  lang, dataset_path_synth, batch_size, train_id):
        dataset_path_real = Path(dataset_path_real)
        if dataset_path_synth is not None:
            dataset_path_synth = Path(dataset_path_synth)
        if dataset_path_reject is not None:
            dataset_path_reject = Path(dataset_path_reject)
        self.train_id = train_id
        self.seed = seed
        self.lang = lang
        self.batch_size = batch_size
        self.rng = random.Random(self.seed)
        self.stats = {}
        self.sample_rate = sample_rate
        self.reject_window_dim = 3 #seconds
        self.train_frac, self.val_frac, self.test_frac, = train_frac, val_frac, test_frac
        self.exe_path = Path(get_curr_dir(__file__)).joinpath("ffmpeg.exe")
        assert self.train_frac + self.val_frac + self.test_frac == 1
        self.db_train, self.db_val, self.db_test = self.prepare_command_dataset(dataset_path_real=dataset_path_real,
                                                                 dataset_path_synth=dataset_path_synth,
                                                                 dataset_path_reject=dataset_path_reject)
        self.write_manifest(db_train=self.db_train, db_val=self.db_val, db_test=self.db_test)
        self.noise_path_base = Path(get_curr_dir(__file__)).joinpath("fsd") #Aggiustare il path
        self.noise_data = self.noise_path_base.joinpath("noise_set")
        self.noise_df = self.create_noise_db()
        self.noise_convert()
        self.noise_split_df()
        self.write_noise_manifest()

    @property
    def train_set(self):
        return self.db_train.copy()
    @property
    def validation_set(self) -> pd.DataFrame:
        return self.db_val.copy()
    @property
    def test_set(self):
        return self.db_test.copy()
    @property
    def noise_train_set(self):
        return self.noise_df_train.copy()
    @property
    def noise_validation_set(self):
        return self.noise_df_val.copy()
    @property
    def noise_test_set(self):
        return self.noise_df_test.copy()
    @property
    def class_weight_train(self):
        df = self.train_set
        labels = list(df.groupby("cmd_index").groups.keys())
        return compute_class_weight("balanced", classes=labels, y=df["cmd_index"].to_numpy())
    @property
    def class_weight_validation(self):
        df = self.validation_set
        labels = list(df.groupby("cmd_index").groups.keys())
        return compute_class_weight("balanced", classes=labels, y=df["cmd_index"].to_numpy())

    '''
    The noise train set, noise validation set and noise test set are saved as csv files.
    '''
    def write_noise_manifest(self):
        def write_file(db: pd.DataFrame, dataset_type):
            assert dataset_type == "train" or dataset_type == "val" or dataset_type == "test"
            name = "validation" if dataset_type == "val" else dataset_type
            fname = f"noise_{name}_{self.lang}_manifest.csv"
            db.to_csv(write_path.joinpath(fname))

        for noise_set, type_set in zip([self.noise_train_set, self.noise_validation_set, self.noise_test_set], ["train", "val", "test"]):
            write_path = Path(get_curr_dir(__file__)).joinpath("manifests")
            write_file(noise_set, type_set)

    '''
    The train set, validation set and test set are saved as json file compatible with the Nvidia Nemo framework.
    Them are saved also as csv files
    '''
    def write_manifest(self, db_train, db_val, db_test):
        def write_file(db: pd.DataFrame, dataset_type):
            assert dataset_type == "train" or dataset_type == "val" or dataset_type == "test"
            name = "validation" if dataset_type == "val" else dataset_type
            fil = open(write_path.joinpath(f"{name}_{self.lang}_manifest.json"), "w")
            for index, row in db.iterrows():
                data = {"audio_filepath": str(row["path"]), "duration": 0.0, "command": row["cmd_index"]}
                json.dump(data, fil)
                fil.write("\n")
            fil.close()
            db.to_csv(write_path.joinpath(f"{name}_{self.lang}_manifest.csv"))

        write_path = Path(get_curr_dir(__file__)).joinpath("manifests")
        write_file(db_train, "train")
        write_file(db_val, "val")
        write_file(db_test, "test")

    '''
    This function use the ffmpeg software to convert the audio file given as input via its path into WAV format
    :param input_audio: path to input audio file
    :param output_audio: path to save the file
    '''
    def convert_audio(self, input_audio, output_audio):
        cmd = [self.exe_path, "-y", "-i", input_audio, output_audio]
        null = subprocess.DEVNULL
        process = subprocess.run(cmd, stdin=null, stdout=null)
        if process.returncode != 0:
            raise Exception("Conversion error for file:", input_audio)

    '''
    This function explores the dataset and splits it into train, test and validation.
    The train set is composed by both real and synthetic speakers, the validation set and test set are composed 
    by only real speakers.
    
    :return: returns the three sets
    '''
    def prepare_command_dataset(self, dataset_path_real, dataset_path_reject, dataset_path_synth, debug=True):
        def split_speakers_list(speakers_db):
            speakers_train, speakers_val_test = train_test_split(speakers_db, train_size=self.train_frac, random_state=self.seed)
            x = self.val_frac / (self.val_frac + self.test_frac)
            speakers_val, speakers_test = train_test_split(speakers_val_test, train_size=x, random_state=self.seed)
            return speakers_train, speakers_val, speakers_test
        def get_analyzator(root, service):
            def real_analyzator(filname: str):
                speaker = Path(root).parent.name
                cmd_index = filname.split("_")[1].split(".")[0]
                return speaker, cmd_index
            if service == "real":
                return real_analyzator
            else:
                 return select_analyzator(service)
        def get_database(speakers_db, speakers_list) -> pd.DataFrame:
            db = pd.DataFrame(columns=speakers_db.columns)
            for speaker in speakers_list:
                data_samples = speakers_db.loc[speakers_db["speaker_id"] == speaker]
                db = db.append(data_samples, ignore_index=True)
            if "cmd_index" in db.columns.tolist():
                db["cmd_index"] = db["cmd_index"].astype("int32")
            return db
        def compute_cmds_sum(db, dataset_split):
            assert dataset_split == "train" or dataset_split == "val" or dataset_split == "test"
            cmd_list = db.groupby("cmd_index").groups
            for k in cmd_list.keys():
                cmd_list[k] = len(cmd_list[k])
            cmd_sum = np.array(list(cmd_list.values())).sum()
            self.stats[f"{dataset_split}_cmds_sum"] = cmd_sum
        def database_analyzer(db: pd.DataFrame, dataset_type: str):
            db["cmd_index"] = db["cmd_index"].astype("int32")
            db = db.sort_values(by="cmd_index")
            db_cmd_index = db.groupby("cmd_index")
            print("Analysis of dataset:", dataset_type)
            for k, v in db_cmd_index.groups.items():
                print(f"{k}:\t{len(v)}")
        def walk(dataset_path):
            speakers_db = pd.DataFrame(columns=["speaker_id", "service", "path", "cmd_index"])
            for root, dirs, files in os.walk(dataset_path, topdown=False, followlinks=True):
                lang_folder = Path(root).name
                if len(files) == 0 or lang_folder != self.lang:
                    continue
                service = Path(root).parent.parent.name if dataset_path == dataset_path_synth else "real"
                analyzator = get_analyzator(root, service)
                for file in files:
                    if file.split(".")[1] != "wav": continue
                    speaker, cmd_index = analyzator(file)
                    path = Path(root).joinpath(file)
                    speakers_db = speakers_db.append({"speaker_id": speaker, "service": service,
                                                                    "path": path, "cmd_index": cmd_index}, ignore_index=True)
            speakers_db["cmd_index"] = speakers_db["cmd_index"].astype("int32")
            return speakers_db
        def convert(audio_path: str):
            fname = Path(audio_path).name
            fname = fname.split(".")[0] + ".wav"
            out_path = Path(audio_path).parent.joinpath(fname)
            if not out_path.exists():
                self.convert_audio(audio_path, out_path)
            return out_path
        def check_exist(df: pd.DataFrame, dataset_path_reject_lang):
            indexes = []
            for index, row in df.iterrows():
                audio_path = row["path"]
                path = Path(dataset_path_reject_lang).joinpath("clips", audio_path)
                if path.exists():
                    indexes.append(index)
            db_out = df.iloc[indexes]
            return db_out
        def get_reject_db(dataset_path_reject):
            assert len(commands.command_eng) == len(commands.command_ita)
            assert self.lang == "ita" or self.lang == "eng"
            dataset_path_reject_lang = Path(dataset_path_reject).joinpath(self.lang)
            reject_db = pd.read_csv(dataset_path_reject_lang.joinpath("train.tsv"), sep="\t")
            reject_db = check_exist(reject_db, dataset_path_reject_lang)
            reject_db = reject_db.loc[:, ["client_id", "path"]]
            reject_db: pd.DataFrame = reject_db.rename(columns={"client_id":"speaker_id"})
            reject_db["path"] = reject_db["path"].apply(lambda x: str(Path(dataset_path_reject_lang).joinpath("clips", x)))
            speakers_reject = list(reject_db.groupby("speaker_id").groups.keys())
            speakers_reject_split = split_speakers_list(speakers_reject)
            db_list = []
            if self.train_id is not None:
                out_dir = Path(dataset_path_reject).joinpath(f"out/train_id_{self.train_id}")
            else:
                out_dir = Path(dataset_path_reject).joinpath("out")
            if out_dir.exists():
                if out_dir.is_symlink():
                    root, dirs, files = next(os.walk(out_dir))
                    for file in files:
                        os.remove(Path(root).joinpath(file))
                    for dir in dirs:
                        shutil.rmtree(Path(root).joinpath(dir))
                else:
                    shutil.rmtree(out_dir)
            out_dir.mkdir(exist_ok=True, parents=True)
            for dataset_type, speakers_list in zip(["train", "val", "test"], speakers_reject_split):
                db = get_database(reject_db, speakers_list)
                db = db.sample(n=int(self.stats[f"{dataset_type}_cmds_sum"]), random_state=self.seed)
                db["path"] = db["path"].apply(lambda x: convert(x))
                db["path"] = self._parallel_cut(db, out_dir, dataset_type)
                db["cmd_index"] = len(commands.command_eng)
                db["cmd_index"] = db["cmd_index"].astype("int32")
                db_list.append(db)
            return tuple(db_list)

        speakers_db_real = walk(dataset_path_real)
        speakers_train, speakers_val, speakers_test = split_speakers_list(list(speakers_db_real.groupby("speaker_id").groups.keys()))
        db_train = get_database(speakers_db_real, speakers_train)
        db_val = get_database(speakers_db_real, speakers_val)
        db_test = get_database(speakers_db_real, speakers_test)
        if dataset_path_synth is not None:
            speakers_db_synth = walk(dataset_path_synth)
            db_train = db_train.append(speakers_db_synth, ignore_index=True)
        compute_cmds_sum(db_train, "train")
        compute_cmds_sum(db_val, "val")
        compute_cmds_sum(db_test, "test")
        if dataset_path_reject is not None:
            reject_dbs = get_reject_db(dataset_path_reject)
            db_train = db_train.append(reject_dbs[0], ignore_index=True)
            db_val = db_val.append(reject_dbs[1], ignore_index=True)
            db_test = db_test.append(reject_dbs[2], ignore_index=True)
        #Shuffle
        db_train = db_train.sample(frac=1, random_state=self.seed)
        if debug:
            database_analyzer(db_train, dataset_type="train")
            database_analyzer(db_val, dataset_type="validation")
            database_analyzer(db_test, dataset_type="test")
        return db_train, db_val, db_test

    def _parallel_cut(self, db: pd.DataFrame, out_dir, dataset_type):
        def cut(audio_path, out_path: Path, index):
            fname = Path(audio_path).name
            out_path = out_path.joinpath(fname)
            audio = AudioSegment.from_file(audio_path, target_sr=self.sample_rate)
            diff = audio.duration - self.reject_window_dim
            if diff > 0:
                start_time = self.rng.uniform(0, diff)
                end_time = start_time + self.reject_window_dim
                audio.subsegment(start_time, end_time)
            soundfile.write(out_path, audio.samples, samplerate=audio.sample_rate, format="WAV")
            return out_path, index
        assert dataset_type in ["train", "val", "test"]
        N_CPU = 7
        parallel = Parallel(n_jobs=N_CPU, prefer="threads")
        func_list = []
        db_out = pd.DataFrame(columns=["path"])
        for index, row in db.iterrows():
            path = row["path"]
            func  = delayed(cut)(path, out_dir, index)
            func_list.append(func)
        res = parallel(func_list)
        for e in res:
            path, index = e
            data = pd.DataFrame(columns=["path"], index=[index], data=[path])
            db_out = db_out.append(data, verify_integrity=True)
        if dataset_type == "test":
            gc.collect()
            get_reusable_executor().shutdown(wait=True, kill_workers=True)
        return db_out["path"]

    def batch_balancing(self, train_set: pd.DataFrame):
        def log(batch):
            if DEBUG:
                for k, v in batch.groupby("cmd_index").groups.items():
                    print(f"{k}:\t{len(v)}", end="\t")
                print()
                print("*"*50)
        DEBUG = True
        train_set_org = train_set.copy()
        train_set_org.reset_index(drop=True, inplace=True)
        train_set_out = pd.DataFrame(columns=train_set_org.columns)
        weights = compute_sample_weight("balanced", train_set_org["cmd_index"].to_numpy())
        rng = torch.Generator()
        rng = rng.manual_seed(self.seed)
        weight_sampler = WeightedRandomSampler(weights, len(train_set_org), replacement=True, generator=rng)
        batch_sampler = BatchSampler(weight_sampler, self.batch_size, drop_last=True)
        batch_list = list(batch_sampler)
        for batch_indexes in batch_list:
            batch: pd.DataFrame = train_set_org.iloc[batch_indexes]
            log(batch)
            train_set_out = train_set_out.append(batch, ignore_index=True)
        return train_set_out

    '''
    Creates the database containing the path to noise file
    The files must be split in folders, the folder defines the category to which it belongs the noise sample
    '''
    def create_noise_db(self)-> pd.DataFrame:
        db = pd.DataFrame(columns=["path", "fname", "noise_type"])
        for root, dirs, files in os.walk(self.noise_data, topdown=False, followlinks=True):
            if len(files) == 0:
                continue
            dir_name = os.path.basename(root)
            for e in files:
                if e in EXCLUDES: continue
                db = db.append({"path": str(Path(root).joinpath(e)), "fname": e, "noise_type": dir_name}, ignore_index=True)
        return db

    '''
    Checks if there are noise samples not in WAV format, then converts them
    '''
    def noise_convert(self):
        flag = False
        for index, row in self.noise_df.iterrows():
            fname = row["fname"]
            if fname in EXCLUDES:
                continue
            path = row["path"]
            if fname.split(".")[1] != "wav":
                flag = True
                out_path = Path(os.path.dirname(path)).joinpath(f"{fname.split('.')[0]}.wav")
                self.convert_audio(path, out_path)
                audio, sr = librosa.load(out_path)
                os.remove(path)
                sd.write(out_path, audio, sr, format="WAV")
        if flag:
            self.noise_df = self.create_noise_db()

    '''
    Returns the noise category containing into noise set
    '''
    def _get_occurency(self) -> pd.DataFrame:
        return self.noise_df["noise_type"].value_counts()

    '''
    Splits the noise set into train set, validation set and test set
    There is no overlap of samples in each set
    '''
    def noise_split_df(self):
        self.noise_df_test = pd.DataFrame(columns=self.noise_df.columns)
        self.noise_df_val = pd.DataFrame(columns=self.noise_df.columns)
        self.noise_df_train = pd.DataFrame(columns=self.noise_df.columns)
        occs = self._get_occurency().index.tolist()
        for occ in occs:
            db: pd.DataFrame = self.noise_df.loc[self.noise_df["noise_type"] == occ]
            db = db.reset_index(drop=True).copy()
            try:
                train_index, val_test_index = train_test_split(db.index.to_numpy(), train_size=self.train_frac, random_state=self.seed)
                x = self.val_frac / (self.val_frac + self.test_frac)
                val_index, test_index = train_test_split(db.iloc[val_test_index].index.to_numpy(), train_size=x, random_state=self.seed)
            except ValueError:
                print(f"{occ} not enough samples")
                continue
            db_train, db_val, db_test = db.iloc[train_index].copy(), db.iloc[val_index].copy(), db.iloc[test_index].copy()
            self.noise_df_test = self.noise_df_test.append(db_test, ignore_index=True)
            self.noise_df_val = self.noise_df_val.append(db_val, ignore_index=True)
            self.noise_df_train = self.noise_df_train.append(db_train, ignore_index=True)
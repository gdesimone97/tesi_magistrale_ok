import os
from pathlib import Path
from utils import command_ita, command_eng
import pandas as pd
import math
import warnings
import subprocess
import shutil

path_out_base = Path("dataset/Dataset_real_out")
path_in_base = Path("dataset/Dataset_real")
db_path = Path("db_dataset_real.csv")
os.environ["PATH"] = os.environ["PATH"] + ";" + r"C:\MIE CARTELLE\PROGRAMMAZIONE\GITHUB\tesi_magistrale\bin"
warnings.filterwarnings("ignore")
TIMEOUT = 2.5
SPEED = 1.8

def myplay(audiopath):
    with open(os.devnull, "w") as devnull:
        cmd = ["ffplay.exe", "-nodisp", "-autoexit", '-af', f'atempo={SPEED}', audiopath]
        subprocess.run(cmd, stdout=devnull, stderr=devnull)

def get_columns() -> list:
    columns = []
    for k in range(2):
        lang = "eng" if k <=0 else "ita"
        for cmd_index in range(0, len(command_ita)):
            s = f"{lang}_{cmd_index}"
            columns.append(s)
    assert len(columns) == len(command_eng) * 2
    return columns

def init_db() -> pd.DataFrame:

    def create_db():
        db = pd.DataFrame(columns=columns)
        for userid in os.listdir(path_in_base):
            data = pd.DataFrame(columns=columns, index=[userid])
            db = db.append(data)
        db["done"] = False
        db.to_csv(db_path, index=True)
        return db

    def load_db():
        assert len(command_eng) == len(command_ita)
        db = pd.read_csv(db_path, index_col=0)
        flag_save = False
        if len(db.columns) - 1 != len(get_columns()):
            flag_save = True
            done_column = db["done"].copy()
            del db["done"]
            nw_columns = list(set(get_columns()) - set(db.columns))
            for c in nw_columns:
                db[c] = [float("nan")]*len(db)
            db["done"] = done_column
        index = db.index.tolist()
        for userid in os.listdir(path_in_base):
            if not path_in_base.joinpath(userid).is_dir() or int(userid) in index:
                continue
            data = pd.DataFrame(columns=columns, index=[userid])
            data["done"] = False
            db = db.append(data)
            flag_save = True
        db = db.astype("O")
        if flag_save:
            db.to_csv(db_path, index=True)
        return db

    path_out_base.mkdir(parents=True, exist_ok=True)
    assert path_in_base.exists()
    columns = get_columns()
    if not db_path.exists():
        db = create_db()
    else:
        db = load_db()
    return db


class Checker:
    def __init__(self, db: pd.DataFrame):
        self.db = db
        for userid in self.db.index.tolist():
            self.userid = userid
            self._check_complete()
        self._filter_db()

    def _filter_db(self):
        self.db_filter = self.db.loc[self.db["done"] == False]

    def _get_audio_file(self):
        for index, row in self.db_filter.iterrows():
            self.userid = index
            for cmd, value in row.iteritems():
                if not math.isnan(value): continue
                lang, cmd_index = cmd.split("_")[0],  cmd.split("_")[1]
                self.lang = lang
                self.cmd = cmd
                self.cmd_index = cmd_index
                audio_path = path_in_base.joinpath(str(self.userid), lang, f"{cmd}.ogg")
                yield audio_path

    def _check_complete(self):
        row: pd.Series = self.db.loc[self.userid].copy()
        del row["done"]
        res: pd.Series = row.isin([float("nan")])
        for index, value in res.iteritems():
            res.at[index] = not value
        done = True if res.all() else False
        self.db.at[self.userid, "done"] = done

    def check(self):
        for audiopath in self._get_audio_file():
            self.audiopath = audiopath
            if not audiopath.exists():
                # self._set_mistake()
                continue
            myplay(str(audiopath))
            self._handler_user_result()
            self._check_complete()
            self.db.to_csv(db_path)

    def copy(self):
        i = 0
        db = self.db.copy()
        db = db.replace(float("nan"), False)
        for userid, row in db.iterrows():
            i += 1
            print(f"\rCopy of {i} of {len(self.db)}", end="")
            del row["done"]
            for cmd, value in row.iteritems():
                if value == False:
                    continue
                userid = str(userid)
                split = cmd.split("_")
                filname = f"{cmd}.ogg"
                lang, cmd_index = split[0], split[1]
                path_out_folder = path_out_base.joinpath(userid, lang)
                path_in_folder = path_in_base.joinpath(userid, lang)
                path_out = path_out_folder.joinpath(filname)
                path_in = path_in_folder.joinpath(filname)
                if path_out.exists(): continue
                path_out_folder.mkdir(parents=True, exist_ok=True)
                shutil.copy(path_in, path_out, follow_symlinks=True)

    def count(self):

        def _count_total():
            out_columns = ["#speaker", "#cmd_eng", "#cmd_ita"]
            db = pd.DataFrame({out_columns[0]: [0], out_columns[1]: [0], out_columns[2]: [0]})
            db_copy = self.db.copy().replace(float("nan"), False)
            for userid, row in db_copy.iterrows():
                del row["done"]
                if row.any():
                    db.at[0, out_columns[0]] += 1
                cmd_eng: pd.Series = row[columns_eng].loc[row[columns_eng] == True]
                cmd_ita: pd.Series = row[columns_ita].loc[row[columns_ita] == True]
                num_cmd_eng = cmd_eng.sum()
                num_cmd_ita = cmd_ita.sum()
                db.at[0, out_columns[1]] += num_cmd_eng
                db.at[0, out_columns[2]] += num_cmd_ita
            path = path_out_base.joinpath("info.csv")
            db.to_csv(str(path))

        def _count_command(lang):
            command_list = command_eng if lang == "eng" else command_ita
            columns_type = columns_eng if lang == "eng" else columns_ita
            command_list = list(map(lambda x: x[1], command_list.items()))
            db_out = pd.DataFrame(columns=command_list)
            db = self.db[columns_type].copy()
            db = db.replace(float("nan"), False)
            db = db.replace(False, 0)
            db = db.replace(True, 1)
            final_sum = db.sum().tolist()
            db_out = db_out.append(dict(zip(command_list, final_sum)), ignore_index=True)
            db_out = db_out.T
            path = path_out_base.joinpath(f"db_{lang}.csv")
            db_out.to_csv(path)

        columns = get_columns()
        columns_eng = columns[:len(command_eng)]
        columns_ita = columns[len(command_eng):]
        _count_total()
        _count_command("eng")
        _count_command("ita")


    def _handler_user_result(self):
        command_type = command_eng if self.lang == "eng" else command_ita
        string_out = command_type[int(self.cmd_index)]
        while True:
            s = f"{self.userid} - {self.cmd} - {string_out}: "
            res = input(s)
            if res == "y" or res == "":
                self._set_ok()
                break
            elif res == 'r':
                myplay(self.audiopath)
            else:
                self._set_mistake()
                break

    def _set_ok(self):
        self.db.at[self.userid, self.cmd] = True

    def _set_mistake(self):
        self.db.at[self.userid, self.cmd] = False

if __name__ == "__main__":
    db = init_db()
    print("Init complete")
    checker = Checker(db)
    checker.check()
    checker.copy()
    # checker.count()
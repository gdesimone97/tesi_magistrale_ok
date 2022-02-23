import os
from pathlib import Path
import pandas as pd
from check_dataset import get_columns
from utils import command_eng, command_ita
import json

class Count:

    def __init__(self):
        self.database = self.init_db()

    def init_db(self):
        columns = get_columns()
        db = pd.DataFrame(columns=columns)
        for userid in os.listdir(path_base):
            if not path_base.joinpath(userid).is_dir(): continue
            data = pd.DataFrame(columns=columns, index=[userid])
            db = db.append(data)
            db = db.replace(float("nan"), False)
            for folder_lang in os.listdir(path_base.joinpath(userid)):
                if not path_base.joinpath(userid, folder_lang).is_dir(): continue
                for fil in os.listdir(path_base.joinpath(userid, folder_lang)):
                    cmd = fil.split(".")[0]
                    db.at[userid, cmd] = True if cmd in columns else False
        return db

    def get_lang_columns(self, lang):
        if lang == "eng":
            return get_columns()[:len(command_eng)]
        else:
            return get_columns()[len(command_eng):]

    def get_lang_command(self, lang):
        command = command_eng if lang == "eng" else command_ita
        return list(map(lambda x: x[1], command.items()))

    def _count_total(self):
        def inner_count_cmd(lang):
            inner_columns = self.get_lang_columns(lang)
            db_inner = db[inner_columns]
            inner_sum = db_inner.sum(axis=1)
            final_sum = inner_sum.sum()
            return final_sum
        def count_speaker():
            speakers_eng_count = 0
            speakers_ita_count = 0
            db_speaker: pd.Series = db.any(axis=1)
            for userid, _ in db_speaker.iteritems():
                path = Path(f"dataset/Dataset_real/{userid}/info.json")
                info = json.load(open(path))
                lang_selected = info["lang"]
                if lang_selected == "both":
                    speakers_eng_count += 1
                    speakers_ita_count += 1
                elif lang_selected == "eng":
                    speakers_eng_count += 1
                elif lang_selected == "ita":
                    speakers_ita_count += 1
                else:
                    raise Exception(f"Error in lang selected with {path}")
            return speakers_eng_count, speakers_ita_count

        out_columns = ["#speaker_eng", "#speaker_ita", "#cmds_eng", "#cmds_ita"]
        db = self.database.copy()
        db_out = pd.DataFrame(columns=out_columns)
        num_speaker_eng, num_speaker_ita = count_speaker()
        db = db.replace(False, 0)
        db = db.replace(True, 1)
        num_command_eng = inner_count_cmd("eng")
        num_command_ita = inner_count_cmd("ita")
        db_out.at[0, out_columns[0]] = num_speaker_eng
        db_out.at[0, out_columns[1]] = num_speaker_ita
        db_out.at[0, out_columns[2]] = num_command_eng
        db_out.at[0, out_columns[3]] = num_command_ita
        db_out.to_csv(path_base.joinpath("info.csv"))

    def _count_command(self):
        db = self.database.copy()
        db = db.replace(False, 0)
        db = db.replace(True, 1)
        final_sum: pd.Series = db.sum()
        db_out = pd.DataFrame(columns=get_columns())
        eng_command = self.get_lang_command("eng")
        ita_command = self.get_lang_command("ita")
        command = eng_command + ita_command
        db_out = db_out.append(dict(zip(command, final_sum)), ignore_index=True)
        db_eng = db_out[eng_command].astype(int)
        db_ita = db_out[ita_command].astype(int)
        db_eng.T.to_csv(path_base.joinpath("db_eng.csv"))
        db_ita.T.to_csv(path_base.joinpath("db_ita.csv"))

    def count(self):
        self._count_total()
        self._count_command()

if __name__ == "__main__":
    exec(open("check_dataset.py").read())
    path_base = Path("dataset/Dataset_real_out")
    count = Count()
    count.count()
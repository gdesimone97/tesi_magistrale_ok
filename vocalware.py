import requests
from utils import command_eng, command_ita
from pathlib import Path
import hashlib
import time

params_list = [
    "EID",
    "LID",
    "VID",
    "TXT",
    "ACC",
    "API",
    "CS"
]

SECRET_KEY = ""
ACCOUNT_ID = ""
API_ID = ""

def md5(params: dict):
    input_data = ""
    for i in range(len(params_list) -1):
        input_data += str(params[params_list[i]])
    input_data += SECRET_KEY
    return hashlib.md5(input_data.encode("utf-8")).hexdigest()

def send_request(params: dict):
    time.sleep(1)
    url = r"http://www.vocalware.com/tts/gen.php"
    res = requests.get(url, params=params)
    if res.status_code != 200:
        time.sleep(5)
        i = 0
        while res.status_code != 200:
            res = requests.get(url, params=params)
            i += 1
            if i > 6 and res.status_code != 200:
                raise Exception(f"Http error: {res.status_code}")
            time.sleep(1)
    return res.content

if __name__ == "__main__":
    engine_eng = [(2, 11), (3, 8), (7, 7)]
    engine_ita = [(2, 10), (3, 2), (7, 2)]
    path_base = Path("voices/vocalware")
    speaker = 0
    cmd_count = 0
    def create(lang):
        global speaker
        global cmd_count
        commands = command_ita if lang == "ita" else command_eng
        lang_id = 7 if lang == "ita" else 1
        engine_lang = engine_ita if lang == "ita" else engine_eng
        for e in engine_lang:
            engine = e[0]
            voice_id_list = range(1, e[1]+1)
            for voice_id in voice_id_list:
                speaker += 1
                for index, cmd in commands.items():
                    filname = f"engine{engine}_voice{voice_id}_cmd{index}.mp3"
                    path = path_base.joinpath(lang, filname)
                    cmd_count += 1
                    if path.exists(): continue
                    params = {
                        params_list[0]: engine,
                        params_list[1]: lang_id,
                        params_list[2]: voice_id,
                        params_list[3]: cmd,
                        params_list[4]: ACCOUNT_ID,
                        params_list[5]: API_ID
                    }

                    params[params_list[6]] = md5(params)
                    audio = send_request(params)
                    with open(path, "wb") as fil:
                        fil.write(audio)

    create("eng")
    create("ita")
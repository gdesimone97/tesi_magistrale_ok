import json

from telegram.ext import Updater
from pathlib import Path
from utils import get_curr_dir, State, send_command
import os
from strings import ITA_STR
import math
import logging

TOKEN = r"1905358159:AAHLRxV694LmBhtb6GH3aqCoG9vc3v4nd7Y" #release bot
#TOKEN = r"2002657157:AAGnJ_0MbmkDahrnkuvGvraIVKmTfDAQBdk" #test bot

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def check_done(state):
    database = state.database
    tot = len(database.loc[database["done_eng"] == True]) + len(database.loc[database["done_ita"] == True])
    return database.all()["done_eng"] and database.all()["done_ita"], tot

def create_bot():
    updater = Updater(TOKEN)
    # updater.dispatcher.add_error_handler(error_handler)
    return updater

def main4():
    path = get_curr_dir(__file__)
    path = Path(path).joinpath("saves")
    updater = create_bot()
    for folder in os.listdir(path):
        try:
            state = State(folder)
            info_path = path.joinpath(folder, "info.json")
            if check_done(state)[0]: continue
            with open(info_path, "r") as fil:
                data = json.load(fil)
            num = math.floor(check_done(state)[1] / 5)
            old_meme = data["meme"]
            data["meme"] = num
            if num == 0: num = 1
            rang = range(old_meme, num)
            with open(info_path, "w") as fil:
                json.dump(data, fil, indent=4)
            if len(rang) > 0:
                updater.bot.send_message(folder, text="Grazie per il tuo impegno😎\nHai vinto questi premi bonus:")
            for e in rang:
                with open(Path(get_curr_dir(__file__)).joinpath("meme", "ita", f"{e}.jpg"), "rb") as fil:
                    ph = fil.read()
                updater.bot.send_photo(folder, photo=ph)
            msg = ITA_STR[22]
            meme = int(data["meme"]) - 1
            msg = msg.format(meme, state.meme_controller.TOTAL_MEME) + "Continua così, non mollare💪"
            print(folder)
            updater.bot.send_message(folder, text=msg)
            updater.bot.send_message(folder, text=ITA_STR[21])
            send_command(folder, updater.bot, state.get_current_command())
        except Exception:
            print(f"Error - {folder}")
            continue

main4()
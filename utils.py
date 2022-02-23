import traceback
from pathlib import Path
import json
import pandas as pd
from datetime import datetime, timedelta
import random
import time
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import BaseTrigger
import os
import logging
from threading import Event, Timer, Thread, Lock
from strings import command_ita, command_eng, ITA_STR, ENG_STR
import math

assert len(command_eng) == len(command_ita)
logger = logging.getLogger(__name__)

#FUNCS
def get_curr_dir(file):
    path = os.path.abspath(os.path.realpath(os.path.dirname(file)))
    return path

def error_handler(update, context):
    if context is None or context.error is None: return
    user = None if update is None else update.effective_user
    text = f"Problem with: {user} - {datetime.now()} - {context.error}"
    logger.error(text)

def get_age_set():
    age_str = []
    for i in range(9):
        s = fr"{10 + 10 * i}-{20 + 10 * i}"
        age_str.append([s])
    return age_str

'''
Formats a command message
'''
def send_command(userid, bot, cmd):
    cmd = f"<b><u>{cmd}</u></b>"
    bot.send_message(chat_id=userid, text=cmd, parse_mode="HTML")

'''
Formats a message
'''
def send_msg(userid, bot, text: str):
    bot.send_message(chat_id=userid, text=text)

'''
Checks if the server if connected to Internet
'''
def is_connected(updater):
    import requests
    try:
        requests.get("http://google.com")
        updater.start_polling()
        logger.info(f"Connected - {datetime.now()}")
        return True
    except Exception:
        logger.warn(f"No internet connection - {datetime.now()}")
        return False

'''
This class stores all the user info.
When a user starts the conversation, first checks if him already exists on server, using his telegram userid.
If not exists this class creates the user directory on disk and initializes the commands database.
If exists it loads the  database
'''
class State:
    def __init__(self, userid):
        update_controller = UpdateController(self)
        self.id = userid.id if not type(userid) is str else userid
        self.path = Path(fr"{get_curr_dir(__file__)}/saves/{self.id}")
        self.info_path = self.path.joinpath("info.json")
        self.last_lang = "ita"
        self.database = pd.DataFrame({"cmd": range(0, len(command_eng)), "done_eng": [False]*len(command_eng),
                                  "done_ita": [False]*len(command_ita)})
        self.database.set_index("cmd", inplace=True)
        if type(userid) is str:
            self.info = {}
        else:
            self.info = {"name": userid.name, "link": userid.link,
                         "gender": None, "age": None,
                         "cmd_index": None, "cmds_done": 0,"meme": 1,
                         "last_lang": self.last_lang, "last_cmd_date": None, "init_complete": False
                         }
        self._load()
        self.text_controller = TextController(self)
        self.meme_controller = MemeController(self)
        update_controller.update()

    '''
    Sends to user the last conversation message
    '''
    def send_last_msg(self, bot):
        memes = self.meme_controller.get_last_meme()
        bot.send_message(self.id, text=self.text_controller.get_str(22))
        bot.send_photo(self.id, photo=memes[0])
        time.sleep(0.5)
        bot.send_message(self.id, text=self.text_controller.get_str(9))
        bot.send_photo(self.id, photo=memes[1])

    '''
    Gets the string associated to index given as input
    '''
    def get_str(self, index):
        return self.text_controller.get_str(index)

    '''
    Sends a reminder to user
    '''
    def remember(self, bot):
        try:
            self._load()
            init_complete = self.info["init_complete"]
            cmd = self.get_current_command()
            last_date = self.info["last_cmd_date"]
            if not init_complete:
                bot.send_message(chat_id=self.id, text=self.text_controller.get_str(19))
            elif last_date is None or (Scheduler.user_is_sleeping(self) and not self.check_done()):
                self._reset_cmds_done()
                bot.send_message(chat_id=self.id, text=self.text_controller.get_str(20))
                with open(Path(f"{get_curr_dir(__file__)}/meme/sad.jpg"), "rb") as fil:
                    ph = fil.read()
                    bot.send_photo(chat_id=self.id, photo=ph)
                time.sleep(1)
                bot.send_message(chat_id=self.id, text=self.text_controller.get_str(21))
                send_command(self.id, bot=bot, cmd=cmd)
        except Exception as e:
            # print(traceback.format_exc())
            logger.warn(f"error: {e}; in remember for {self.id} - {datetime.now()}")
        return self.id

    '''
    Checks if the bot needs to give the user a break
    '''
    def check_pause(self):
        PAUSE = 6
        n_commands = self.info["cmds_done"]
        if n_commands >= PAUSE:
            self._reset_cmds_done()
            return True
        return False

    '''
    Saves the user info on disk
    '''
    def _save_info(self):
        with open(self.info_path, "w") as fil:
            json.dump(self.info, fil, indent=4)

    '''
    Updates the last language and associated command index prompted to the user.
    Then save the information to disk
    '''
    def _save_index_lang(self, index):
        self.info["cmd_index"] = index
        self.info["last_lang"] = self.last_lang
        self._save_info()

    '''
    Sets to 0 the pause count
    '''
    def _reset_cmds_done(self):
        self.info["cmds_done"] = 0
        self._save_info()

    '''
    Updates the pause count and the date of the last command done.
    Then saves the info on disk
    '''
    def _save_commands_count_last_date(self):
        self.info["cmds_done"] = self.info["cmds_done"] + 1
        now = datetime.now()
        self.info["last_cmd_date"] = str(now)
        self._save_info()

    '''
    Saves the commands database on disk
    '''
    def _save_database(self):
        self.database.to_csv(self.path.joinpath("database.csv"))

    '''
    Checks if user has done all commands
    '''
    def check_done(self):
        lang = self.info["lang"]
        assert lang == "both" or lang == "eng" or lang == "ita"
        if lang == "both":
            res = self.database.all()["done_eng"] and self.database.all()["done_ita"]
        elif lang == "eng":
            res = self.database.all()["done_eng"]
        elif lang == "ita":
            res = self.database.all()["done_ita"]
        return res

    '''
    Loads the commands database and the user info
    '''
    def _load(self) -> dict:
        if self.info_path.exists():
            with open(self.info_path) as fil:
                self.info = json.load(fil)
                self.last_lang = self.info["last_lang"]
                self.database = pd.read_csv(self.path.joinpath("database.csv"), index_col=0)
        else:
            self.path.mkdir(parents=True, exist_ok=False)
            with open(self.info_path, "w") as fil:
                json.dump(self.info, fil, indent=4)
            self._save_database()
    '''
    Returns the set lang
    '''
    @property
    def lang(self):
        return self.info["lang"]

    '''
    Returns the total number of commands
    '''
    @property
    def commands_len(self):
        assert len(command_ita) == len(command_eng)
        return len(command_ita)*2 if self.lang == "both" else len(command_ita)

    '''
    Returns the number done by the user
    '''
    @property
    def cmds_done(self):
        done_eng = len(self.database.loc[self.database["done_eng"] == True])
        done_ita = len(self.database.loc[self.database["done_ita"] == True])
        lang = self.info["lang"]
        if lang == "eng": return done_eng
        elif lang == "ita": return done_ita
        else:
            db_eng: pd.DataFrame = self.database.loc[self.database["done_eng"] == True]
            db_ita: pd.DataFrame = self.database.loc[self.database["done_ita"] == True]
            done = len(db_eng) + len(db_ita)
            return done

    '''
    Returns the index of the next command to asks to user.
    If the user has not completed the command stored in its info file then this function returns the associate index
    '''
    def get_next_command(self, lang):
        def error_handler():
            db_eng = self.database.loc[self.database["done_eng"] == False]
            db_ita = self.database.loc[self.database["done_ita"] == False]
            if len(db_eng) != 0:
                self.last_lang = "eng"
                index = min(db_eng.index)
            else:
                self.last_lang = "ita"
                index = min(db_ita.index)
            return index

        last_lang = self.info["last_lang"]
        if lang == "ita":
            command_set: pd.Series = self.database.loc[self.database["done_ita"] == False]
            self.last_lang = "ita"
        elif lang == "eng":
            command_set: pd.Series = self.database.loc[self.database["done_eng"] == False]
            self.last_lang = "eng"
        else:
            if last_lang == "eng":
                self.last_lang = "ita"
                command_set: pd.Series = self.database.loc[self.database["done_ita"] == False]
            else:
                self.last_lang = "eng"
                command_set: pd.Series = self.database.loc[self.database["done_eng"] == False]
        try:
            cmd_index = random.sample(command_set.index.tolist(), 1)[0]
        except ValueError:
            cmd_index = error_handler()
            logger.warn(f"ValueError for {self.id} selected index command: {cmd_index} for {self.last_lang} - {datetime.now()}")
        self._save_index_lang(cmd_index)
        return cmd_index

    '''
    Returns the string associate to index command to ask to user
    '''
    def get_current_command(self):
        cmd_index = self.info["cmd_index"]
        last_lang = self.info["last_lang"]
        lang = self.info["lang"]
        if lang == "ita":
            col_value = "done_ita"
        elif lang == "eng":
            col_value = "done_eng"
        else:
            col_value = "done_eng" if last_lang == "eng" else "done_ita"
        if cmd_index is None or self.database._get_value(cmd_index, col_value):
            cmd_index = self.get_next_command(self.info["lang"])
        return self.text_controller.get_command_str(cmd_index, self.last_lang)

    '''
    Sends the next meme to user
    '''
    def get_meme(self, bot):
        def send(photo):
            bot.send_message(chat_id=self.id, text=self.text_controller.get_str(22))
            bot.send_photo(chat_id=self.id, photo=photo)
            time.sleep(2)
            bot.send_message(chat_id=self.id, text=self.text_controller.get_str(23))

        meme = self.meme_controller.get_meme()
        if meme is not None:
            send(meme)

    '''
    Returns the number of commands done by the user. the START and STOP commands are counted only once
    '''
    def _get_command_count_meme(self):
        done_eng = len(self.database.loc[self.database["done_eng"] == True])
        done_ita = len(self.database.loc[self.database["done_ita"] == True])
        lang = self.info["lang"]
        if lang == "eng": return done_eng
        elif lang == "ita": return done_ita
        else:
            db_eng: pd.DataFrame = self.database.loc[self.database["done_eng"] == True]
            db_ita: pd.DataFrame = self.database.loc[self.database["done_ita"] == True]
            done = len(db_eng) + len(db_ita)
            try:
                if db_eng.at[18, "done_eng"]: #Attention here !!!!
                    done -= 1
            except KeyError:
                pass
            try:
                if db_eng.at[19, "done_eng"]: #Attention here !!!!
                    done -= 1
            except KeyError:
                pass
            return done

    '''
    Sets to True the init_complete key into info file. Then saves it on disk
    '''
    def set_init_complete(self):
        self.info["init_complete"] = True
        self._save_info()

    '''
    Saves on disk an audio file
    '''
    def save(self, audio):
        def save_duplicates(index):
            if index == 18 or index == 19:
                path_eng = self.path.joinpath("eng")
                path_ita = self.path.joinpath("ita")
                path_eng.mkdir(parents=True, exist_ok=True)
                path_ita.mkdir(parents=True, exist_ok=True)
                audio.download(path_eng.joinpath(fr"eng_{index}.ogg"))
                audio.download(path_ita.joinpath(fr"ita_{index}.ogg"))
                self.database._set_value(index, "done_eng", True)
                self.database._set_value(index, "done_ita", True)
                self._save_commands_count_last_date()
                self._save_database()
                return True
            return False

        lang = self.info["last_lang"]
        index = self.info["cmd_index"]
        assert lang == "eng" or lang == "ita"
        assert index is not None
        if save_duplicates(index): return
        path = self.path.joinpath("eng") if lang == "eng" else self.path.joinpath("ita")
        path.mkdir(parents=True, exist_ok=True)
        audio.download(path.joinpath(fr"{lang}_{index}.ogg"))
        if lang == "eng":
            self.database._set_value(index, "done_eng", True)
        else:
            self.database._set_value(index, "done_ita", True)
        self._save_database()
        self._save_commands_count_last_date()

'''
This class implements a scheduler. Its job is to remind the user to complete the data collection
'''
class Scheduler:
    class Trigger(BaseTrigger):
        def __init__(self, delay):
            self.delay = delay
            self.first = True
        def get_next_fire_time(self, previous_fire_time, now):
            if self.first:
                self.first = False
                return now + timedelta(seconds=self.delay)
            else: return  None
    '''
    Builds the scheduler.
    Takes in input the bot instance
    '''
    def __init__(self, bot):
        self.scheduler = BackgroundScheduler()
        self.bot = bot
        self.set_userid = set()
        self.db_path = Path(fr"{get_curr_dir(__file__)}/scheduler.db")
        self.lock = Lock()
        self.scheduler.start()

    '''
    If the user has not done a command or he did not send for long time.
    This function computes the delay from the current datetime to the nearest target datetime
    '''
    def _get_delay(self, state: State):
        last_date = state.info["last_cmd_date"]
        if last_date is None or Scheduler.user_is_sleeping(state):
            now = datetime.now()
            target_date1 = datetime.strptime(fr"{now.year}-{now.month}-{now.day}-08:30", "%Y-%m-%d-%H:%M")
            target_date2 = datetime.strptime(fr"{now.year}-{now.month}-{now.day}-19:30", "%Y-%m-%d-%H:%M")
            diff1 = (target_date1 - now).seconds
            diff2 = (target_date2 - now).seconds
            # delay = min(diff1, diff2)
            delay = diff2
            delay = delay + timedelta(seconds=len(self.set_userid)).total_seconds()
            return delay

    '''
    Returns True if the user does not send commands for a long time
    '''
    @staticmethod
    def user_is_sleeping(state: State):
        last_date = state.info["last_cmd_date"]
        scheduled_hours = Scheduler.get_scheduled_hour(state)
        assert scheduled_hours is not None
        assert last_date is not None
        last_date = datetime.strptime(last_date, "%Y-%m-%d %H:%M:%S.%f")
        scheduled_hours = timedelta(seconds=float(scheduled_hours))
        now = datetime.now()
        diff = now - last_date
        return True if diff >= scheduled_hours else False

    '''
    Computes the delay before to start to send reminders
    '''
    @staticmethod
    def set_scheduled_hour(state):
        hour = timedelta(hours=random.randint(15, 50)).total_seconds()
        path = Path(get_curr_dir(__file__)).joinpath("saves", str(state.id), "scheduler.json")
        data = {"scheduled_hour": str(hour)}
        with open(path, "w") as fil:
            json.dump(data, fil, indent=4)

    @staticmethod
    def get_scheduled_hour(state):
        path = Path(get_curr_dir(__file__)).joinpath("saves", state.id, "scheduler.json")
        with open(path) as fil:
            data = json.load(fil)
            return data["scheduled_hour"]

    '''
    Schedules the user reminder
    '''
    def check(self):
        curr_dir = get_curr_dir(__file__)
        path = Path(fr"{curr_dir}/saves/")
        path.mkdir(parents=True, exist_ok=True)
        for directory in os.listdir(path):
            userid = directory
            state = State(userid)
            if state.check_done(): continue
            delay = self._get_delay(state)
            if delay is not None and userid not in self.set_userid:
                self.lock.acquire()
                self.add(delay, state)
                Scheduler.set_scheduled_hour(state)
                self.lock.release()
        log = f"scheduled: {str(self.set_userid)} - {datetime.now()}"
        logger.info(log)

    '''
    Function decorator.
    Removes a user by the reminder set
    '''
    def _decorator(self, func):
        def wrapper(**kwargs):
            userid = func(**kwargs)
            self.lock.acquire()
            self.set_userid.remove(userid)
            self.lock.release()
            logger.info(f"Removing: {str(userid)}")
        return wrapper

    '''
    Adds a user to reminder set
    '''
    def add(self, delay, state: State):
        self.set_userid.add(state.id)
        self.scheduler.add_job(self._decorator(state.remember), trigger=self.Trigger(delay), kwargs={"bot": self.bot},
                               timezone="Europe/Rome")
        logger.info(f"Adding: {state.id} at {datetime.now() + timedelta(seconds=delay)}")

'''
This class implement a Python Timer.
Checks if a user needs of a reminder every time interval
'''
class SchedulerTimer(Thread):

    def __init__(self, interval, bot):
        Thread.__init__(self)
        self.scheduler = Scheduler(bot)
        self.interval = interval
        self.function = self._decorator(self.scheduler.check)
        self.event = Event()
        self.timer: Timer = None
        self.stop_flag = False

    def _decorator(self, function):
        def wrapper():
            function()
            self.event.set()
        return wrapper

    def stop(self):
        if self.timer is not None:
            self.timer.cancel()
        logger.info(f"Shutdown scheduler - {datetime.now()}")
        self.scheduler.scheduler.shutdown()
        self.stop_flag = True
        self.event.set()

    def run(self):
        self.function()
        while not self.stop_flag:
            self.event.clear()
            self.timer = Timer(self.interval, self.function)
            self.timer.start()
            self.event.wait()

'''
This class allows to developer to update user info when he needs to modify the bot behaviour.
In this ways the developers can modify the user info from one version of the bot to another.
'''
class UpdateController:
    def __init__(self, state: State):
        self.state = state

    def _update_scheduled_hours(self):
        if not Path(fr"{get_curr_dir(__file__)}/saves/{self.state.id}/scheduler.json").exists():
            Scheduler.set_scheduled_hour(self.state)

    def _update_lang(self):
        if self.state.info.get("lang") is None:
            self.state.info["lang"] = "both"
            self.state._save_info()

    def _update_database(self):
        assert len(command_eng) == len(command_ita)
        if len(self.state.database) < len(command_ita):
            diff = list(set(command_eng.keys()) - set(self.state.database.index))
            diff.sort()
            if "cmd" in self.state.database.columns.tolist():
                self.state.database.set_index("cmd", inplace=True)
            for e in diff:
                data = pd.DataFrame(data=[[False, False]], columns=self.state.database.columns, index=[e])
                self.state.database = self.state.database.append(data, verify_integrity=True)
        self.state._save_database()

    def _update_meme_count(self):
        cmds_done = self.state.cmds_done
        if cmds_done < 0: cmds_done = 0
        meme_id = math.ceil(cmds_done / self.state.meme_controller.reward_freq)
        if cmds_done == self.state.commands_len:
            meme_id = self.state.meme_controller.TOTAL_MEME +1
        elif meme_id == 0:
            meme_id = 1
        elif cmds_done % self.state.meme_controller.reward_freq == 0:
            meme_id += 1
        self.state._reset_cmds_done()
        self.state.info["meme"] = meme_id
        self.state._save_info()

    def update(self):
        self.state._load()
        self._update_scheduled_hours()
        self._update_lang()
        self._update_database()
        self._update_meme_count()

'''
This class implements the rewards system
'''
class MemeController:

    def __init__(self, state: State):
        self.state = state
        self.base_path = Path(get_curr_dir(__file__)).joinpath("meme")
        self.REWARD_FREQ = 5 # A reward every 5 commands

    '''
    Returns the number of total meme respect to selected lang
    '''
    @property
    def TOTAL_MEME(self):
        lang = self.state.info["lang"]
        assert lang == "ita" or lang == "eng" or lang == "both"
        return 11 if lang == "both" else 6

    '''
    Returns the next reward
    '''
    def get_meme(self):
        lang = self.state.info["lang"]
        path_end = "eng" if lang == "eng" else "ita"
        path = self.base_path.joinpath(path_end)
        cmds_count = self.state._get_command_count_meme()
        meme = self.state.info["meme"]
        if cmds_count >= int(meme) * self.reward_freq and meme < self.TOTAL_MEME:
            self.state.info["meme"] += 1
            self.state._save_info()
            with open(path.joinpath(f"{meme}.jpg"), "rb") as fil:
                return fil.read()
        return None

    @property
    def reward_freq(self):
        return self.REWARD_FREQ

    '''
    Returns the last meme
    '''
    def get_last_meme(self):
        lang = self.state.info["lang"]
        path_end = "eng" if lang == "eng" else "ita"
        path = self.base_path.joinpath(path_end)
        with open(path.joinpath(r"thk.jpg"), "rb") as fil:
            meme2 = fil.read()
        with open(path.joinpath(f"{str(self.state.info['meme'])}.jpg"), "rb") as fil:
            meme1 = fil.read()
        self.state.info["meme"] += 1
        self.state._save_info()
        return meme1, meme2

'''
This implements the communication system based on text.
Returns the required strings respect to selected lang.
'''
class TextController:
    def __init__(self, state: State):
        self.state = state

    '''
    Checks if the string contains a reward message, if yes formats it
    '''
    def _level_check(self, indx, msg: str):
        lv = int(self.state.info["meme"]) - 1
        tot = self.state.meme_controller.TOTAL_MEME
        if indx == 22:
            msg = msg.format(lv, tot) + self._get_single_str(24)
        return msg

    '''
    Returns a string that does not require formatting
    '''
    def _get_single_str(self, indx):
        lang = self.state.info["lang"]
        assert lang == "ita" or lang == "eng" or lang == "both"
        if lang == "ita" or lang == "both":
            msg = ITA_STR[indx]
        elif lang == "eng":
            msg = ENG_STR[indx]
        return msg

    '''
    Returns the string associated to index
    '''
    def get_str(self, indx):
        msg = self._get_single_str(indx)
        return self._level_check(indx, msg)

    '''
    Returns the command string associated to command index and selected lang 
    '''
    def get_command_str(self, cmd_indx, lang):
        assert lang == "ita" or lang == "eng"
        return command_ita[cmd_indx] if lang == "ita" else command_eng[cmd_indx]
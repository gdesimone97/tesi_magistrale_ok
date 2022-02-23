from telegram.ext import CallbackContext, ConversationHandler
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from utils import State
from utils import get_age_set, send_command, get_curr_dir
import numpy as np
import logging
from datetime import datetime
import time
from pathlib import Path
from strings import WELCOME_STR
import shutil

LANG_STATE, GENDER_STATE, AGE_STATE, START, STATE, PAUSE = 0, 1, 2, 3, 4, 5
logger = logging.getLogger(__name__)

'''
Asks to user to choose the commands lang. After sets the state to GENDER_STATE
'''
def get_lang(update: Update, context: CallbackContext):
    update.callback_query.answer()
    state: State = context.user_data["state"]
    lang = update.callback_query.data.split("_")[1]
    state.info["lang"] = lang
    state._save_info()
    keyword = [[InlineKeyboardButton(state.get_str(15), callback_data="gender_Male"),
                InlineKeyboardButton(state.get_str(16), callback_data="gender_Female")]]
    time.sleep(1)
    context.bot.send_message(state.id, state.get_str(6), reply_markup=InlineKeyboardMarkup(keyword))
    return GENDER_STATE

'''
Asks the gender of the user. After sets the state to AGE_STATE
'''
def get_gender(update: Update, context: CallbackContext):
    state: State = context.user_data["state"]
    gender = update.callback_query.data.split("_")[1]
    state.info["gender"] = gender
    state._save_info()

    age_set = get_age_set()
    for i, e in enumerate(age_set):
        age_set[i] = InlineKeyboardButton(str(e[0]), callback_data=f"age_{e}")
    age_set = np.array(age_set)
    age_set = age_set.reshape((-1, 3))
    keyboard = InlineKeyboardMarkup(age_set)

    time.sleep(1)

    context.bot.send_message(state.id, state.get_str(0), reply_markup=keyboard)
    update.callback_query.answer()
    return AGE_STATE

'''
Asks the age of the user. After starts to send the commands
'''
def get_age(update: Update, context: CallbackContext):
    state: State = context.user_data["state"]
    query = update.callback_query
    age = query.data.split("_")[1]
    state.info["age"] = age
    state._save_info()
    state.set_init_complete()

    time.sleep(1)
    context.bot.send_message(state.id, state.get_str(1))
    time.sleep(5)
    context.bot.send_message(state.id, state.get_str(2))
    query.answer()
    time.sleep(1.5)
    return start_conversation(update, context)

'''
This callback is classed when the user start the bot (/start command).
Checks if the user has already started the bot: if no, starts a new conversation else continues the previous conversation.
If the user has already done the data collection terminates the conversation
'''
def start(update: Update, context: CallbackContext) -> int:

    keyword = [[InlineKeyboardButton("Entrambe - Both", callback_data="lang_both")],
                [InlineKeyboardButton("English", callback_data="lang_eng"),
                InlineKeyboardButton("Italiano", callback_data="lang_ita")]]

    state = State(update.effective_user)
    context.user_data["state"] = state

    log_text = '{' + fr"{update.effective_user.full_name}, {update.effective_user.link}, {update.effective_user.id}, complete: {state.check_done()}" + '}'
    logger.info(fr"Entra: {log_text} - {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

    if state.check_done():
        update.message.reply_text(state.get_str(8))
        return ConversationHandler.END

    if state.info["gender"] != None:
        if state.info["age"] == None:
            age_set = get_age_set()
            for i, e in enumerate(age_set):
                age_set[i] = InlineKeyboardButton(str(e[0]), callback_data=f"age_{e}")
            age_set = np.array(age_set)
            age_set = age_set.reshape((-1, 3))
            keyboard = InlineKeyboardMarkup(age_set)
            context.bot.send_message(state.id, state.get_str(18), reply_markup=keyboard)
            return AGE_STATE
        else:
            state: State = context.user_data["state"]
            state.set_init_complete()
            cmd = state.get_current_command()
            context.bot.send_message(chat_id=state.id, text=state.get_str(18))
            time.sleep(1)
            send_command(state.id, bot=context.bot, cmd=cmd)
            return STATE

    update.message.reply_text(WELCOME_STR[0])
    time.sleep(6)
    update.message.reply_text(WELCOME_STR[1])
    time.sleep(3)
    update.message.reply_text(WELCOME_STR[2], reply_markup=InlineKeyboardMarkup(keyword))
    return LANG_STATE

'''
Asks a command to user
'''
def start_conversation(update: Update, context: CallbackContext):
    state: State = context.user_data["state"]
    cmd = state.get_current_command()
    send_command(state.id, bot=context.bot, cmd=cmd)
    return STATE

'''
Continues a previous conversation
'''
def restart_conversation(update: Update, context: CallbackContext):
    state = State(update.effective_user)
    context.user_data["state"] = state

    if state.info["cmd_index"] is None:
        update.message.reply_text(state.get_str(7))
        return ConversationHandler.END
    if state.check_done():
        update.message.reply_text(state.get_str(8))
        return ConversationHandler.END
    audio = update.message.voice.get_file()
    state.save(audio)
    if state.check_done():
        state.send_last_msg(context.bot)
        return ConversationHandler.END
    cmd = state.get_current_command()
    send_command(state.id, bot=context.bot, cmd=cmd)
    return STATE

'''
Stores an audio file sent by user. Can asks to user if he wants to take a break
'''
def receive(update: Update, context: CallbackContext):
    state: State = context.user_data["state"]
    state.save(update.message.voice.get_file())
    if state.check_done():
        state.send_last_msg(context.bot)
        return ConversationHandler.END
    state.get_meme(context.bot)
    if state.check_pause():
        keyword = [[InlineKeyboardButton(state.get_str(11), callback_data="pause_y"), InlineKeyboardButton(state.get_str(12), callback_data="pause_n")]]
        update.message.reply_text(state.get_str(10), reply_markup=InlineKeyboardMarkup(keyword))
        return PAUSE
    cmd = state.get_current_command()
    send_command(state.id, bot=context.bot, cmd=cmd)
    return STATE

'''
Read the user input and start or not a break
'''
def conv_pause(update: Update, context: CallbackContext):
    state: State = context.user_data["state"]
    query = update.callback_query
    cmd = state.get_current_command()
    if query.data == "pause_y":
        context.bot.send_message(chat_id=state.id, text=state.get_str(13))
        time.sleep(1.5)
        send_command(state.id, bot=context.bot, cmd=cmd)
        query.answer()
        return ConversationHandler.END
    elif query.data == "pause_n":
        context.bot.send_message(chat_id=state.id, text=state.get_str(14))
        time.sleep(1)
        send_command(state.id, bot=context.bot, cmd=cmd)
        query.answer()
        return STATE

'''
If the conversation ended but the user did not complete the task, 
this callback continues the conversation when the user make a query using the Inline keyboard
'''
def select_query(update: Update, context: CallbackContext):
    query_data = update.callback_query.data
    state = State(update.effective_user)
    context.user_data["state"] = state
    identifier, data = query_data.split("_")[0], query_data.split("_")[1]
    if identifier == "pause":
        return conv_pause(update, context)
    elif identifier == "gender" and state.info["gender"] is None:
        return get_gender(update, context)
    elif identifier == "age" and state.info["age"] is None:
        return get_age(update, context)
    elif identifier == "lang" and state.info["gender"] is None:
        return get_lang(update, context)
    else:
        update.callback_query.answer()
        return start(update, context)

'''
Deletes all user files
'''
def reset(update: Update, context: CallbackContext):
    userid = update.effective_user["id"]
    path = Path(get_curr_dir(__file__)).joinpath("saves", str(userid))
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    update.message.reply_text("Sistema ripristinato\nPer ricominciare scrivi: /start")
    logger.info(f"user {userid} has resetted profile - {datetime.now()}")
    return ConversationHandler.END

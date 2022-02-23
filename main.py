#doc: https://python-telegram-bot.readthedocs.io/en/stable/
#github rep: https://github.com/python-telegram-bot/python-telegram-bot

import logging
import time
import telegram.error
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler, CallbackQueryHandler
from callbacks import start, restart_conversation, receive, get_gender, get_age, conv_pause, select_query, get_lang, reset, \
    GENDER_STATE, AGE_STATE, STATE, PAUSE, LANG_STATE
from utils import get_curr_dir, SchedulerTimer, error_handler, is_connected
import os
from datetime import datetime, timedelta

TOKEN = r"" #Telegram bot token

def main():

    def create_bot():
        updater = Updater(TOKEN)
        updater.dispatcher.add_handler(conv_handler)
        updater.dispatcher.add_handler(CommandHandler('reset', reset))
        updater.dispatcher.add_error_handler(error_handler)
        return updater

    os.chdir(get_curr_dir(__file__))

    # Enabling logger
    logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logging.getLogger('apscheduler.scheduler').setLevel(logging.ERROR)
    logging.getLogger('apscheduler.executors.default').setLevel(logging.ERROR)
    logging.getLogger('telegram.vendor.ptb_urllib3.urllib3.connectionpool').setLevel(logging.ERROR)

    states = {}
    states[AGE_STATE] = [CallbackQueryHandler(get_age)]
    states[STATE] = [MessageHandler(Filters.voice, receive)]
    states[GENDER_STATE] = [CallbackQueryHandler(get_gender)]
    states[PAUSE] = [CallbackQueryHandler(conv_pause)]
    states[LANG_STATE] = [CallbackQueryHandler(get_lang)]

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start),
                      MessageHandler(Filters.voice, restart_conversation),
                      CallbackQueryHandler(select_query)
                      ],
        states=states,
        fallbacks=[CommandHandler('reset', reset)],
        conversation_timeout=None
    )

    # Start the Bot
    # Create the Updater and pass it your bot's token.
    updater = create_bot()
    logger.info(datetime.now().strftime("%d-%m-%Y %H:%M"))

    try:
        updater.start_polling()
    except telegram.error.NetworkError:
        while not is_connected(updater) :
            updater = create_bot()
            time.sleep(60)

    #Start timer
    TIME_CHECK = 45 #minutes
    scheduler_timer = SchedulerTimer(timedelta(minutes=TIME_CHECK).total_seconds(), updater.bot)

    scheduler_timer.start()

    # Run the bot until the user presses Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT
    updater.idle()
    scheduler_timer.stop()

if __name__ == "__main__":
    main()



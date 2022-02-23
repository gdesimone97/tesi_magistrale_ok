from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from pathlib import Path
from utils import command_ita, command_eng

voices_eng = [
"en-US_AllisonV3Voice",
"en-US_EmilyV3Voice",
"en-US_HenryV3Voice",
"en-US_KevinV3Voice",
"en-US_LisaV3Voice",
"en-US_MichaelV3Voice",
"en-US_OliviaV3Voice",
"en-GB_CharlotteV3Voice",
"en-GB_JamesV3Voice",
"en-GB_KateV3Voice",
"en-AU_MadisonVoice",
"en-AU_CraigVoice",
]
voices_ita = ["it-IT_FrancescaV3Voice"]

def test():

    with open('hello_world.wav', 'wb') as audio_file:
        audio_file.write(
            text_to_speech.synthesize(
                'Hello world',
                voice='en-US_AllisonV3Voice',
                accept='audio/wav'
            ).get_result().content)

def create(voice, path):
    global cmd_count
    lang = path.split("/")[1]
    if lang == "eng":
        cmds_list = command_eng
    elif lang == "ita":
        cmds_list = command_ita
    else:
        raise Exception

    for k, v in cmds_list.items():
        voice_name = voice.split("-")[1].split("_")[1]
        path_to_save = path_base.joinpath(path, fr"{voice_name}_{k}.wav")
        cmd_count += 1
        if path_to_save.exists(): continue
        with open(path_to_save, 'wb') as audio_file:
            audio_file.write(
                text_to_speech.synthesize(
                    v,
                    voice=voice,
                    accept='audio/wav'
                ).get_result().content)

if __name__ == "__main__":
    speaker = len(voices_ita) + len(voices_eng)
    cmd_count = 0
    authenticator = IAMAuthenticator('')
    text_to_speech = TextToSpeechV1(authenticator=authenticator)
    text_to_speech.set_service_url('https://api.eu-de.text-to-speech.watson.cloud.ibm.com/instances/e317c863-b016-4e4f-9fc9-d1122ea9abb8')
    path_base = Path("./voices/ibm")
    print("#Eng neural voices:", len(voices_eng),"\n#Italian neural voices:", len(voices_ita))
    for voice in voices_eng:
        create(voice, "neural/eng")
    for voice in voices_ita:
        create(voice, "neural/ita")
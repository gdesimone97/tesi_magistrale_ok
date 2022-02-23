import os

from azure.cognitiveservices.speech.speech import SpeechConfig, SpeechSynthesizer
from azure.cognitiveservices.speech.audio import AudioOutputConfig
from utils import command_eng, command_ita
from pathlib import Path
import time

KEY1 = r""
KEY2 = r""

region = r"westeurope"

neural_voices_eng = [
"en-AU-NatashaNeural",
"en-AU-WilliamNeural",
"en-CA-ClaraNeural",
"en-CA-LiamNeural",
"en-HK-YanNeural",
"en-HK-SamNeural",
"en-IN-NeerjaNeural",
"en-IN-PrabhatNeural",
"en-IE-EmilyNeural",
"en-IE-ConnorNeural",
"en-NZ-MollyNeural",
"en-NZ-MitchellNeural",
"en-PH-RosaNeural",
"en-PH-JamesNeural",
"en-SG-LunaNeural",
"en-SG-WayneNeural",
"en-ZA-LeahNeural",
"en-ZA-LukeNeural",
"en-GB-LibbyNeural",
"en-GB-MiaNeural",
"en-GB-RyanNeural",
"en-US-AriaNeural",
"en-US-JennyNeural",
"en-US-GuyNeural",
"en-US-AmberNeural",
"en-US-AshleyNeural",
"en-US-CoraNeural",
"en-US-ElizabethNeural",
"en-US-MichelleNeural",
"en-US-MonicaNeural",
"en-US-AnaNeural",
"en-US-BrandonNeural",
"en-US-ChristopherNeural",
"en-US-JacobNeural",
"en-US-EricNeural"
]

standard_voices_eng = [
"en-AU-HayleyRUS",
"en-CA-HeatherRUS",
"en-CA-Linda",
"en-IN-Heera",
"en-IN-PriyaRUS",
"en-IN-Ravi",
"en-IE-Sean",
"en-GB-George",
"en-GB-HazelRUS",
"en-GB-Susan",
"en-US-BenjaminRUS",
"en-US-GuyRUS",
"en-US-AriaRUS",
"en-US-ZiraRUS"
]
neural_voices_ita = ["it-IT-ElsaNeural", "it-IT-IsabellaNeural", "it-IT-DiegoNeural"]
standard_voices_ita = ["it-IT-Cosimo", "it-IT-LuciaRUS"]


def test():
    speech_config = SpeechConfig(subscription=KEY1, region=region)
    speech_config.speech_synthesis_voice_name = "en-AU-HayleyRUS"
    audio_config = AudioOutputConfig(filename="voices/file2.wav")

    synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    synthesizer.speak_text("A simple test to write to a file.")

def main():

    def create(voice, path):
        global speaker
        global cmd_count
        speaker += 1
        lang = path.split("/")[1]
        if lang == "eng":
            cmds_list = command_eng
        elif lang == "ita":
            cmds_list = command_ita
        else:
            raise Exception

        for fil in os.listdir(path_base.joinpath(path)):
            if path_base.joinpath(path, fil).stat().st_size == 0:
                os.remove(path_base.joinpath(path, fil))

        for k, v in cmds_list.items():
            voice_name = voice.split("-")[2]

            #Neural voices
            path_to_save = path_base.joinpath(path)
            fil_path = path_to_save.joinpath(fr"{voice_name}_{k}.wav")
            cmd_count += 1
            if fil_path.exists(): continue
            audio_config = AudioOutputConfig(filename=str(fil_path))
            time.sleep(1)
            speech_config.speech_synthesis_voice_name = voice
            synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
            synthesizer.speak_text(v)
            time.sleep(1)

            if fil_path.stat().st_size == 0:
                print("To restart for", path)
                continue

    speech_config = SpeechConfig(subscription=KEY1, region=region)
    path_base = Path(r"voices/azure/")

    print("\nneural eng")
    for i, voice in enumerate(neural_voices_eng):
        create(voice, r"neural/eng")
        print(f"\r{i+1} of {len(neural_voices_eng)}", end='')

    print("\nstandard eng")
    for i, voice in enumerate(standard_voices_eng):
        # create(voice, r"standard/eng")
        print(f"\r{i + 1} of {len(standard_voices_eng)}", end='')

    print("\nneural ita")
    for i, voice in enumerate(neural_voices_ita):
        create(voice, r"neural/ita")
        print(f"\r{i + 1} of {len(neural_voices_ita)}", end='')

    print("\nstandard ita")
    for i, voice in enumerate(standard_voices_ita):
        # create(voice, r"standard/ita")
        print(f"\r{i + 1} of {len(standard_voices_ita)}", end='')


speaker = 0
cmd_count = 0

if __name__ == "__main__":
    main()
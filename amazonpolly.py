from utils import command_eng, command_ita
from pathlib import Path
from boto3 import Session

voices_eng = {
    "Nicole": "standard",
    "Olivia": "neural",
    "Russell": "standard",
    "Amy": "both",
    "Emma": "both",
    "Brian": "both",
    "Aditi": "standard",
    "Raveena": "standard",
    "Ivy": "both",
    "Joanna": "both",
    "Kendra": "both",
    "Kimberly": "both",
    "Salli": "both",
    "Joey": "both",
    "Justin": "both",
    "Kevin": "neural",
    "Matthew": "both",
    "Geraint": "standard"
}

voices_ita = {
    "Carla": "standard",
    "Bianca": "standard",
    "Giorgio": "standard"
}

AWS_ACCESS_KEY = ""
AWS_SECRET_KEY = ""

session = Session(aws_access_key_id=AWS_ACCESS_KEY,
                      aws_secret_access_key=AWS_SECRET_KEY,
                      region_name="eu-west-2")
polly = session.client("polly")

def mytest():
    response = polly.synthesize_speech(VoiceId='Joanna',
                                              OutputFormat='mp3',
                                              Text='This is a sample text to be synthesized.',
                                              Engine='neural')

    file = open('speech.mp3', 'wb')
    file.write(response['AudioStream'].read())
    file.close()

speaker = 0
cmd_count = 0

def main():

    def create(voice, lang, voice_type):
        global speaker
        global cmd_count
        speaker += 1
        command_list = command_eng if lang == "eng" else command_ita
        if voice_type == "both":
            voice_type = ["neural", "standard"]
        elif voice_type == "standard":
            voice_type = ["standard"]
        else:
            voice_type = ["neural"]
        for index, cmd in command_list.items():
            for engine in voice_type:
                filname = f"{voice}_{index}.mp3"
                path_to_save = path_base.joinpath(engine, lang, filname)
                cmd_count += 1
                if path_to_save.exists(): continue
                audio = polly.synthesize_speech(VoiceId=voice,
                                                      OutputFormat='mp3',
                                                      Text=cmd,
                                                      Engine=engine)['AudioStream'].read()

                with open(path_to_save, "wb") as fil:
                    fil.write(audio)

    path_base = Path("voices").joinpath("polly")
    for k, v in voices_eng.items():
        create(k, "eng", v)
    for k, v in voices_ita.items():
        create(k, "ita", v)


if __name__ == "__main__":
    main()
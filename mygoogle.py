from google.cloud import texttospeech
import os
from utils import command_ita, command_eng
from pathlib import Path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\MIE CARTELLE\PROGRAMMAZIONE\GITHUB\tesi_magistrale\keys\google_key.json"

neural_voices_eng = [
"en-AU-Wavenet-A",
"en-AU-Wavenet-B",
"en-AU-Wavenet-C",
"en-AU-Wavenet-D",
"en-IN-Wavenet-A",
"en-IN-Wavenet-B",
"en-IN-Wavenet-C",
"en-IN-Wavenet-D",
"en-GB-Wavenet-A",
"en-GB-Wavenet-B",
"en-GB-Wavenet-C",
"en-GB-Wavenet-D",
"en-GB-Wavenet-F",
"en-US-Wavenet-A",
"en-US-Wavenet-B",
"en-US-Wavenet-C",
"en-US-Wavenet-D",
"en-US-Wavenet-E",
"en-US-Wavenet-F",
"en-US-Wavenet-G",
"en-US-Wavenet-H",
"en-US-Wavenet-I",
"en-US-Wavenet-J"
]

standard_voices_eng= [
"en-AU-Standard-B",
"en-AU-Standard-C",
"en-AU-Standard-D",
"en-IN-Standard-A",
"en-IN-Standard-B",
"en-IN-Standard-C",
"en-IN-Standard-D",
"en-GB-Standard-A",
"en-GB-Standard-B",
"en-GB-Standard-C",
"en-GB-Standard-D",
"en-GB-Standard-F",
"en-US-Standard-A",
"en-US-Standard-B",
"en-US-Standard-C",
"en-US-Standard-D",
"en-US-Standard-E",
"en-US-Standard-F",
"en-US-Standard-G",
"en-US-Standard-H",
"en-US-Standard-I",
"en-US-Standard-J"
]

neural_voices_ita = [
"it-IT-Wavenet-A ",
"it-IT-Wavenet-B ",
"it-IT-Wavenet-C ",
"it-IT-Wavenet-D "
]

standard_voices_ita = [
"it-IT-Standard-A",
"it-IT-Standard-B",
"it-IT-Standard-C",
"it-IT-Standard-D"
]


def test():
    """Synthesizes speech from the input string of text or ssml.

    Note: ssml must be well-formed according to:
        https://www.w3.org/TR/speech-synthesis/
    """


    # Instantiates a client
    client = texttospeech.TextToSpeechClient()

    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text="Hello, World!")

    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # The response's audio_content is binary.
    with open("output.mp3", "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')

def main():
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
            voice_name = voice.split("-")
            voice_name = f"{voice_name[1]}_{voice_name[2]}_{voice_name[3]}"
            path_to_save = path_base.joinpath(path)
            path_to_save.mkdir(parents=True, exist_ok=True)
            fil_path = path_to_save.joinpath(fr"{voice_name}_cmd{k}.mp3")
            cmd_count += 1
            if fil_path.exists(): continue
            synthesis_input = texttospeech.SynthesisInput(text=v)
            voice_service = texttospeech.VoiceSelectionParams(language_code=voice.split("-")[0], ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL) #TODO
            response = client.synthesize_speech(input=synthesis_input, voice=voice_service, audio_config=audio_config)
            with open(fil_path, "wb") as out:
                out.write(response.audio_content)
                # print('Audio content written to file "output.mp3"')

    client = texttospeech.TextToSpeechClient()
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    path_base = Path(r"voices/google/")

    for voice in neural_voices_eng:
        create(voice, r"neural/eng")

    for voice in standard_voices_eng:
        create(voice, r"standard/eng")

    for voice in neural_voices_ita:
        create(voice, r"neural/ita")

    for voice in standard_voices_ita:
        create(voice, r"standard/ita")

speaker = len(neural_voices_eng) + len(neural_voices_ita) + len(standard_voices_ita) + len(standard_voices_eng)
cmd_count = 0
if __name__ == "__main__":
    main()
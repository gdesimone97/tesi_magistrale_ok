import soundfile as sf
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder
from utils import command_ita, command_eng
from pathlib import Path
import torch
import gc
import time

def mytest():
    # Download and load the pretrained fastpitch model
    spec_generator = SpectrogramGenerator.from_pretrained(model_name="tts_en_fastpitch").cuda()
    # Download and load the pretrained hifigan model
    vocoder = Vocoder.from_pretrained(model_name="tts_hifigan").cuda()

    # All spectrogram generators start by parsing raw strings to a tokenized version of the string
    parsed = spec_generator.parse("You can type your sentence here to get nemo to produce speech.")
    # They then take the tokenized string and produce a spectrogram
    spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
    # Finally, a vocoder converts the spectrogram to audio
    audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

    # Save the audio to disk in a file called speech.wav
    # Note vocoder return a batch of audio. In this example, we just take the first and only sample.
    sf.write("speech.wav", audio.to('cpu').detach().numpy()[0], 22050)

cmd_count = 0
speaker = 0
def main():
    def create(lang):
        global speaker
        global cmd_count
        commands = command_ita if lang == "ita" else command_eng
        for spectro_id in spectro_geneator:
            for vocecoder_id in vocecoder_list:
                speaker += 1
                for index, cmd in commands.items():
                    cmd_count += 1
                    filname = f"{spectro_id.split('_')[2]}_{vocecoder_id.split('_')[1]}_cmd{index}.wav"
                    if path.exists(): continue
                    path = path_base.joinpath("na", lang, filname)
                    try:
                        spectro = SpectrogramGenerator.from_pretrained(model_name=spectro_id).cpu()
                        vocecoder = Vocoder.from_pretrained(model_name=vocecoder_id).cpu()
                        parsed = spectro.parse(cmd)
                        spectrogram = spectro.generate_spectrogram(tokens=parsed)
                        audio = vocecoder.convert_spectrogram_to_audio(spec=spectrogram)
                        # del spectro
                        # del vocecoder
                        # del parsed
                        # del spectrogram
                    except RuntimeError:
                        print("memory end")
                        gc.collect()
                        torch.cuda.empty_cache()
                        time.sleep(3)
                        spectro = SpectrogramGenerator.from_pretrained(model_name=spectro_id)
                        vocecoder = Vocoder.from_pretrained(model_name=vocecoder_id)
                        parsed = spectro.parse(cmd)
                        spectrogram = spectro.generate_spectrogram(tokens=parsed)
                        audio = vocecoder.convert_spectrogram_to_audio(spec=spectrogram)
                    sf.write(path, audio.to('cpu').detach().numpy()[0], SAMPLING_RATE)
                    del audio

    spectro_geneator = ["tts_en_tacotron2", "tts_en_glowtts", "tts_en_fastspeech2", "tts_en_fastpitch"]
    vocecoder_list = ["tts_waveglow_88m", "tts_squeezewave", "tts_uniglow", "tts_melgan", "tts_hifigan"]
    path_base = Path("voices/nemo")
    SAMPLING_RATE = 22050
    create("eng")

if __name__ == "__main__":
    main()
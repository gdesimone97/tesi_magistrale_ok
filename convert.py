from pathlib import Path
import os
import subprocess
from global_utils import get_curr_dir, EXCLUDES

def convert_to_wav(audio_dir: Path, audio_write: Path):

    def walk(path):
        for root, dirs, files in os.walk(path, followlinks=True, topdown=False):
            if len(files) == 0:
                continue
            for fil in files:
                if fil in EXCLUDES: continue
                filname = fil.split(".")[0] + ".wav"
                out = Path(root.replace(audio_dir.name, audio_write.name))
                out.mkdir(parents=True, exist_ok=True)
                if out.joinpath(filname).exists():
                    continue
                p = subprocess.run([exe, "-i", Path(root).joinpath(fil), out.joinpath(filname)])
                if p.returncode != 0:
                    raise Exception(f"Conversion error for {fil}")


    exe = Path(get_curr_dir(__file__)).joinpath("bin/ffmpeg.exe")
    if audio_write.exists():
        print(f"\n{audio_write} already exists")
        # input("Press a key to continue...")

    audio_write.mkdir(parents=True, exist_ok=True)
    walk(audio_dir)

if __name__ == "__main__":
    print("Converting real")
    convert_to_wav(audio_dir=Path(get_curr_dir(__file__)).joinpath("dataset/Dataset_real_out"),
                   audio_write=Path(get_curr_dir(__file__)).joinpath("dataset/Dataset_real_conv"))
    print("Converting synth")
    convert_to_wav(audio_dir=Path(get_curr_dir(__file__)).joinpath("dataset/Dataset_synth"),
                   audio_write=Path(get_curr_dir(__file__)).joinpath("dataset/Dataset_synth_conv"))
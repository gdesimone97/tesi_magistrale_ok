import os

import simpleaudio as sa
from pathlib import Path


def play(filpath):
    wave_obj = sa.WaveObject.from_wave_file(filpath)
    play_obj = wave_obj.myplay()
    play_obj.wait_done()

if __name__ == "__main__":
    path = Path(r"C:\MIE CARTELLE\PROGRAMMAZIONE\GITHUB\tesi_magistrale\voices\nemo\na\eng")

    for fil in os.listdir(path):
        filpath = str(path.joinpath(fil))
        while True:
            play(filpath)
            cmd = input(f"Delete {fil}: ")
            if cmd == "y":
                os.remove(filpath)
                break
            elif cmd == "n":
                break
import os

folder = r"C:\MIE CARTELLE\PROGRAMMAZIONE\GITHUB\tesi_magistrale\dataset\Dataset_synth\naturalreaders\neural\ita"
os.chdir(folder)

voice = "fabiana"
country = "it"

for fil in os.listdir("."):
    split = fil.split(".")[0]
    try:
        cmd_index = int(split)
    except ValueError:
        continue
    name = f"{voice}_{country}_{cmd_index}.mp3"
    if os.path.exists(name):
        raise Exception(f"file: {name} already exists")
    os.rename(fil, name)
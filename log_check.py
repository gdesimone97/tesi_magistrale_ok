
path = r"C:\MIE CARTELLE\PROGRAMMAZIONE\GITHUB\tesi_magistrale\copy_log.txt"
with open(path, encoding='utf-8') as fil:
    lines = fil.readlines()
for line in lines:
    s = line.split("-")
    try:
        if s[1] == " ERROR " or s[1] == " WARNING ":
            print(s)
    except IndexError:
        continue
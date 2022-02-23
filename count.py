
if __name__ == "__main__":
    print("Count synth...")
    exec(open("count_synth.py").read())
    print("Count real...")
    exec(open("count_real.py").read())
    print()
    res = input("Do you want convert? ").lower()
    if res == "y":
        exec(open("convert.py").read())
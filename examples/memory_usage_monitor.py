import subprocess
import time, sys

def get_memory():
    command = ["nvidia-smi"]
    result = subprocess.run(command, capture_output=True, text=True)
    parts = result.stdout.split("|")
    MiB_parts = []
    for part in parts:
        if "MiB" in part:
            MiB_parts.append(part)
    card_zero = MiB_parts[0].split("/")
    card_zero = int(card_zero[0].replace(" ", "").strip("MiB"))

    card_one = MiB_parts[1].split("/")
    card_one = int(card_one[0].replace(" ", "").strip("MiB"))

    return card_zero, card_one



if __name__ == "__main__":
    command = ["python3", "latency_profile.py", "--tensor-parallel-size=2"]
    process = subprocess.Popen(command)
    while process.poll() is None:
        mem0, mem1 = get_memory()
        print(mem0, mem1)
        time.sleep(0.1)

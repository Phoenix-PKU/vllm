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
    command = ["python3", "latency_profile.py", '--max-num-seqs=1']
    if True:
        command.append("--tensor-parallel-size=2")

    process = subprocess.Popen(command)
    memory_list = [[], []]
    """
    while process.poll() is None:
        mem0, mem1 = get_memory()
        # print(mem0, mem1)
        memory_list[0].append(mem0)
        memory_list[1].append(mem1)
        time.sleep(0.1)
    print("max memory of GPU0: ", max(memory_list[0]))
    print("max memory of GPU1: ", max(memory_list[1]))
    """

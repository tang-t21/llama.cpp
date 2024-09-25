"""Microbenchmarking for CPU offloading"""

import argparse
import json
import os
import random
import sys
import numpy as np
import subprocess
import time

if __name__ == "__main__":

    # path_json = "../fiddler/benchmarks/lmsys_chat.jsonl"
    # with open(path_json, "r") as f:
    #     data = [json.loads(line)["conversation"][0]["content"] for line in f]
    # dataset_name="LMSYS"
    # texts = []
    # for d in data:
    #     if len(d) == 0:
    #         continue
    #     # the input of the first round
    #     texts.append(" ".join(d.split()))

    path_json = "../fiddler/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json"
    model_path = "../llama.cpp/models/mixtral-87B-v0.1.gguf"
    with open(path_json, "r") as f:
        data = json.load(f)
    dataset_name = "ShareGPT"
    texts = []
    for d in data:
        if len(d["conversations"]) == 0:
            continue
        # the input of the first round
        texts.append(" ".join(d["conversations"][0]["value"].split()))

    random.seed(0)
    random.shuffle(texts)
    # for input_token in [16, 32, 64, 128]:
    #     for output_token in [16, 32, 64, 128, 256, 512]:
    with open(f"./latency.txt", "a") as f:
        f.write(f"eval on dataset {dataset_name}\n")
        f.write(f"input_token, output_token, prefill_time, decode_time, token/s\n")
    idx_text = 0
    for input_token in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
        random.shuffle(texts)
        input_text = None
        for text in texts:
            if len(text.split()) >= input_token:
                # enough input length
                input_text =  text
                break
        if input_text == None:
            print("No enough length!")
            continue
        for output_token in [64, 128, 256, 512, 1024]:
            print(f"input_token: {input_token}, output_token: {output_token}")
            # print("text:", text)
            subprocess.run(
                [
                    "./build/bin/llama-cli",
                    "-m",
                    model_path,
                    "-p",
                    f'"{input_text}"',
                    "--prompt_length",
                    str(input_token),
                    "-n",
                    str(output_token),
                    "-e",
                    "-ngl",
                    "7",
                    "-t",
                    "8"
                ],
                check=True,  # Optional: will raise an error if the command fails
            )
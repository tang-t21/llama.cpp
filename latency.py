"""Microbenchmarking for CPU offloading"""

import argparse
import json
import os
import random
import sys
import numpy as np
import subprocess
import time

def load_input_data(dataset_name, dataset_path):
    texts = []
    if dataset_name == "LMSYS":
        with open(dataset_path, "r") as f:
            data = [json.loads(line)["conversation"][0]["content"] for line in f]
        for d in data:
            if len(d) == 0:
                continue
            # the input of the first round
            texts.append(" ".join(d.split()))
    elif dataset_name == "ShareGPT":
        with open(dataset_path, "r") as f:
            data = json.load(f)
        for d in data:
            if len(d["conversations"]) == 0:
                continue
            # the input of the first round 
            texts.append(" ".join(d["conversations"][0]["value"].split())) 
    else:
        raise ValueError("No matching dataset")
    return texts

if __name__ == "__main__":

   
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of dataset")
    parser.add_argument("--data_path", type=str, help="Path to dataset")
    args = parser.parse_args()
    print(args.dataset, args.data_path)
    texts = load_input_data(args.dataset, args.data_path)

    model_path = "../old-llama.cpp/models/mixtral-8-7B.gguf"
    random.seed(0)
    random.shuffle(texts)
    # for input_token in [16, 32, 64, 128]:
    #     for output_token in [16, 32, 64, 128, 256, 512]:
    with open(f"./latency.txt", "a") as f:
        f.write(f"eval on dataset {args.dataset}\n")
        f.write(f"input_token, output_token, prefill_time, decode_time, token/s\n")
    idx_text = 0
    # for input_token in [32, 64, 128, 256, 512]:
    for input_token in [1024]:
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
                    "15",
                    "-t",
                    "8",
                    "-fa",
                ],
                check=True,  # Optional: will raise an error if the command fails
            )
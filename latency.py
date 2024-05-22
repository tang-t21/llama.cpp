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

    path_json = "./ShareGPT_V3_unfiltered_cleaned_split.json"
    with open(path_json, "r") as f:
        data = json.load(f)

    texts = []
    for d in data:
        if len(d["conversations"]) == 0:
            continue
        # the input of the first round
        texts.append(" ".join(d["conversations"][0]["value"].split()))

    random.seed(0)
    random.shuffle(texts)
    n_sample = 3
    # for input_token in [16, 32, 64, 128]:
    #     for output_token in [16, 32, 64, 128, 256, 512]:
    with open(f"./latency.txt", "a") as f:
        f.write(f"input_token, output_token, prefill_time, decode_time, token/s\n")
    idx_text = 0
    for input_token in [32, 64, 128, 256, 512]:
        for output_token in [64, 128, 256, 512]:
            while True:
                text = texts[idx_text]
                idx_text += 1
                if len(text.split()) >= input_token:
                    # enough input length
                    break
            total_time_sum = 0
            print(f"input_token: {input_token}, output_token: {output_token}")
            # print("text:", text)
            subprocess.run(
                [
                    "./build/bin/main",
                    "-m",
                    "/mnt/storage/keisuke/weights/Mixtral-8x7B-v0.1/ggml-model-f16.gguf",
                    "-p",
                    f'"{text}"',
                    "-pp",
                    str(input_token),
                    "-n",
                    str(output_token),
                    "-e",
                    "-ngl",
                    "16",
                ],
                check=True,  # Optional: will raise an error if the command fails
            )
            # print(max(model.cpu_expert_time), min(model.cpu_expert_time))
            # print(model.outliner_nums)
            # print(sum(model.outliner_nums), len(model.outliners))
            # print(sum(model.outliners) / len(model.outliners))
            # print(
            #     "Est improved performance:",
            #     prefill_time + decode_time - sum(model.outliners) * 0.88 / 10**6,
            # )
            # print(
            #     f"CPU Layer Num: | {sum(model.cpu_layer_num)/len(model.cpu_layer_num):.2f} | {np.var(model.cpu_layer_num):.2f}"
            # )
            # print(
            #     f"OneToken | {sum(model.one_token_time)/len(model.one_token_time):.2f} ms | {np.var(model.one_token_time):.2f} ms"
            # )
            # print("         | Average value | Variation | Portion")
            # print(
            #     f"CPUExpert | {sum(model.cpu_expert_time)/len(model.cpu_expert_time):.2f} | {np.var(model.cpu_expert_time):.2f} | {sum(model.cpu_expert_time)/(decode_time+prefill_time)/10**6:.2f}"
            # )
            # print(
            #     f"GPUExpert | {sum(model.gpu_expert_time)/len(model.gpu_expert_time):.2f} | {np.var(model.gpu_expert_time):.2f} | {sum(model.gpu_expert_time)/(decode_time+prefill_time)/10**6:.2f}"
            # )
            # print(
            #     f"Attention | {sum(model.attention_time)/len(model.attention_time):.2f} | {np.var(model.attention_time):.2f} | {sum(model.attention_time)/(decode_time+prefill_time)/10**6:.2f}"
            # )
            # print(
            #     f"Selection | {sum(model.selection_time)/len(model.selection_time):.2f} | {np.var(model.selection_time):.2f} | {sum(model.selection_time)/(decode_time+prefill_time)/10**6:.2f}"
            # )
            # print(
            #     f"Optconfig | {sum(model.search_config_time)/len(model.search_config_time):.2f} | {np.var(model.search_config_time):.2f} | {sum(model.search_config_time)/(decode_time+prefill_time)/10**6:.2f}"
            # )

            # # write to file
            # with open(f"./latency.txt", "a") as f:
            #     f.write(
            #         f"{input_token}, {output_token}, {total_time_sum / n_sample:.2f}, {(input_token+output_token) *n_sample/total_time_sum:.2f}\n"
            #     )

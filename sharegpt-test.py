"""Microbenchmarking for CPU offloading"""

import argparse
import json
import os
import random
import sys
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Model path. default `mistralai/Mixtral-8x7B-v0.1`.",
    )
    parser.add_argument(
        "--cpu-offload",
        type=int,
        default=1,
        choices=[0, 1],
        help="0: exeute at GPU (baseline), 1: offload to CPU.",
    )

    parser.add_argument("--beam_width", type=int, default=1, help="Beam search width.")
    parser.add_argument("--torch_threads", type=int, default=16, help="Torch threads.")
    parser.add_argument("--cpp_threads", type=int, default=44, help="C++ threads.")

    args = parser.parse_args()

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
    n_sample = 5

    # for input_token in [16, 32, 64, 128]:
    #     for output_token in [16, 32, 64, 128, 256, 512]:
    for input_token in [16, 32, 64, 128]:
        for output_token in [32, 64, 128]:
            idx_text = 0
            prefill_time_sum, decode_time_sum, hit_rate_sum = 0, 0, 0
            print(f"input_token: {input_token}, output_token: {output_token}")
            for _ in range(n_sample):
                while True:
                    text = texts[idx_text]
                    idx_text += 1
                    if len(text.split()) >= input_token:
                        # enough input length
                        break
                text = text[:input_token]
                # print("text:", text)
                subprocess.call(
                    f"CUDA_VISIBLE_DEVICES=1 ./build/bin/main -m /mnt/storage/keisuke/weights/Mixtral-8x7B-v0.1/ggml-model-f16.gguf -p '{text}' -n {output_token} -e -t 16 -ngl 16",
                    shell=True,
                )
            # write to file
            # with open(
            #     f"./results/latency-{args.torch_threads}-{args.cpp_threads}.txt", "a"
            # ) as f:
            #     f.write(
            #         f"input_token: {input_token}, output_token: {output_token}, "
            #         f"prefill_time: {prefill_time_sum / n_sample}, "
            #         f"decode_time: {decode_time_sum / n_sample}, "
            #         f"hit_rate: {hit_rate_sum / n_sample},"
            #         f"{output_token *n_sample/ (decode_time_sum):.2f}token/s\n"
            #     )

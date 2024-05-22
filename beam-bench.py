import subprocess
import json
import random

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

    # input_lengths = [2**i for i in range(5, 9)]
    # output_lengths = [2**i for i in range(5, 9)]
    input_lengths = [32]
    output_lengths = [64]
    # beam_widths = [4 * i for i in range(1, 5)]
    beam_widths = [1]
    for input_length in input_lengths:
        for output_length in output_lengths:
            for beam_width in beam_widths:
                print(
                    f"Running input_length={input_length}, output_length={output_length}, beam_width={beam_width}"
                )
                idx_text = 0
                while True:
                    text = texts[idx_text]
                    idx_text += 1
                    if len(text.split()) >= input_length:
                        # enough input length
                        break
                subprocess.run(
                    [
                        f"./build/bin/beam-search /mnt/storage/keisuke/weights/Mixtral-8x7B-v0.1/ggml-model-f16.gguf {beam_width} '{text}' {output_length} {input_length}"
                    ],
                    shell=True,
                )

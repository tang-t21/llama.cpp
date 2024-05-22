import subprocess
import json
import random

if __name__ == "__main__":

    input_lengths = [2**i for i in range(5, 9)]
    output_lengths = [2**i for i in range(5, 9)]
    batch_sizes = [4 * i for i in range(1, 5)]
    for input_length in input_lengths:
        for output_length in output_lengths:
            for batch_size in batch_sizes:
                print(
                    f"Running input_length={input_length}, output_length={output_length}, batch_size={batch_size}"
                )
                kv_max = batch_sizes*(input_length+output_length)
                subprocess.run(
                    [
                        f"./build/bin/batched-bench /mnt/storage/keisuke/weights/Mixtral-8x7B-v0.1/ggml-model-f16.gguf {batch_size} {output_length} {input_length}"
                    ],
                    shell=True,
                )
                ./batched-bench MODEL_PATH [N_KV_MAX] [N_BATCH] [N_UBATCH] [IS_PP_SHARED] [NGL] [MMQ] <PP> <TG> <PL>

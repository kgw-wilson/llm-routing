import time
from pathlib import Path
import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from llama_cpp import Llama
from data_processing.constants import (
    FILTERED_WITH_EMBEDDINGS_PATH,
    PROMPT_COL,
    RESULTS_CSV_FILENAME,
)

# Experimental constants so we can have a fair comparison
MAX_TOKENS = 40
TEMPERATURE = 1.0

# Directory for exp8 - (where to save results)
EXP8_PATH = Path(__file__).parent.parent / "exp8"
RESULTS_CSV_PATH = EXP8_PATH / RESULTS_CSV_FILENAME

# Load RouterBench data to use a prompt to test the model
df = pd.read_pickle(FILTERED_WITH_EMBEDDINGS_PATH)

# Create a task string using the first sample
TASK = f"""You are a task analyzer.
Do NOT solve the prompt.
Describe (in English):
1) Task type
2) Difficulty (easy/medium/hard)
3) Skills required

Respond in 1-3 short lines.

Prompt:

{df.iloc[0][PROMPT_COL]}
"""


def _evaluate_transformers_model(results_rows: list[dict]) -> None:

    print("Evaluating Mistral-7B-Instruct-v0.3 model from transformers...")

    # Precision from float32 should help avoid softmax instability and so
    # generate better output. CPU can't use float16 or float16 so PyTorch
    # has to emulate those with float32 they're not used. Disabling
    # double quantization should reduce noise.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
    )
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    load_start = time.time()

    # Force all layers onto the CPU; "auto" and {"": "mps"} both caused a RuntimeError
    # On my m1 macbook. The best device_map may be different for your machine.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": "cpu"},
        quantization_config=bnb_config,
    )

    load_time = time.time() - load_start

    tokenized_input = tokenizer(TASK, return_tensors="pt")
    tokenized_input = {k: v.to(model.device) for k, v in tokenized_input.items()}

    generation_start = time.time()

    # Sampling + quantization amplifies instability, so don't sample
    output = model.generate(
        **tokenized_input,
        max_new_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        do_sample=False,
    )

    generation_time = time.time() - generation_start

    # Find out the number of tokens in the input prompt and slice that out
    # to decode only new tokens
    input_len = tokenized_input["input_ids"].shape[1]
    decoded_output = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)

    results_rows.append(
        {
            "package": "transformers",
            "output": decoded_output,
            "generation_time": generation_time,
            "load_time": load_time,
        }
    )


def _evaluate_llamacpp_model(results_rows: list[dict]) -> None:

    print("Evaluating Quantized Mistral-7B-Instruct-v0.3 model from llama.cpp...")

    parser = argparse.ArgumentParser(description="Run LLM inference experiment")

    parser.add_argument(
        "--llamacpp_model_path",
        type=str,
        required=True,
        help="Path to the GGUF model file for llama.cpp",
    )

    args = parser.parse_args()

    model_path = args.llamacpp_model_path

    load_start = time.time()

    model = Llama(model_path=model_path)

    load_time = time.time() - load_start

    generation_start = time.time()

    # With llama.cpp, the output does not include the input prompt and it
    # handles decoding on its own
    output = model(
        TASK,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    generation_time = time.time() - generation_start

    results_rows.append(
        {
            "package": "llama.cpp",
            "output": output["choices"][0]["text"],
            "generation_time": generation_time,
            "load_time": load_time,
        }
    )

    # Ensures resources are released cleanly, can cause errors otherwise
    model.close()


def _run_experiment() -> None:
    if RESULTS_CSV_PATH.exists():
        print(f"Experiment 8 results already exist at: {RESULTS_CSV_PATH}")
    else:
        results_rows = []

        _evaluate_transformers_model(results_rows)
        _evaluate_llamacpp_model(results_rows)

        # Save results to CSV for easy viewing
        results_df = pd.DataFrame(results_rows)
        print(results_rows)
        results_df.to_csv(RESULTS_CSV_PATH, index=False)


if __name__ == "__main__":
    _run_experiment()

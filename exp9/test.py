import time
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from llama_cpp import Llama
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from data_processing.constants import (
    FILTERED_WITH_EMBEDDINGS_PATH,
    PROMPT_COL,
    RESULTS_CSV_FILENAME,
    embedding_model,
    SAMPLE_ID_COL,
    PROMPT_EMBEDDING_COL,
)
from utils import (
    TEST_SPLIT_SIZE,
    RANDOM_STATE,
    K_VALUES,
    predict_top_k,
    evaluate_predictions,
)

# Experiment constants - determine sample size and allowed length of output
# from LLM. Want enough samples to make a good judgment without too many
# because generation takes time. Want the LLM to be able to finish its
# response without generating a response that's too long and noisy.
# Want output that's deterministic, concise, and consistent, not creative
# or random. This also helps repeatability of results. Context size includes
# the generated tokens, so we need to calculate the maximum amount of prompt
# tokens the model can generate. Special tokens are not considered since
# The mistral model seems to do fine without them.
NUM_SAMPLES = 500
MAX_TOKENS = 80
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = 0
MODEL_CONTEXT_SIZE = 512

# How many samples to do before saving
SAVE_EVERY = 50


# Task description template, should be concise enough for the LLM to
# clearly understand how to produce the desired output while having
# enough detail to generate an output with a descriptive embedding.
TASK_TEMPLATE = """
You are a task analyst.

Summarize what the user is asking for in 1-2 short sentences.
Do NOT repeat or quote the input.
Do NOT solve the task or explain how to do it.
Do NOT include reasoning or examples.

Describe the task itself, not the process for completing it.

If the input is unclear or malformed, give a best-effort high-level description.

USER INPUT:
{prompt}

The task is to
"""


# Directory for exp9 - (where to save results)
EXP9_PATH = Path(__file__).parent.parent / "exp9"
RESULTS_CSV_PATH = EXP9_PATH / RESULTS_CSV_FILENAME
MODEL_OUTPUT_PATH = EXP9_PATH / "model_output.csv"


def _evaluate_llamacpp_model() -> None:

    # Load the Mistral 7b model using command line args to find its path
    parser = argparse.ArgumentParser(description="Run LLM inference experiment")
    parser.add_argument(
        "--llamacpp_model_path",
        type=str,
        required=True,
        help="Path to the GGUF model file for llama.cpp",
    )
    args = parser.parse_args()
    model_path = args.llamacpp_model_path

    print("Loading Quantized Mistral-7B-Instruct-v0.3 model with llama.cpp...")

    # Explicitly set the context longer than what is given by defauly for the quantized model.
    # According to the model metadata 'llama.context_length': '32768'
    # But it seems llama.cpp deftauls to a smaller context length (512) which is smaller
    # Than the longest prompt (max is ~1174 tokens, mean is ~213)
    model = Llama(model_path=model_path, n_ctx=4096)

    # Results rows contains accuracy data and gets saved only at the end
    results_rows = []

    # Record model output for each sample
    model_output_rows = []

    # Load RouterBench data to use a prompt to test the model
    df = pd.read_pickle(FILTERED_WITH_EMBEDDINGS_PATH)

    prompts = [str(TASK_TEMPLATE.format(prompt=p)) for p in df[PROMPT_COL].tolist()]

    sample_ids = df[SAMPLE_ID_COL].tolist()

    task_descriptions = []

    # Generate task descriptions one prompt at a time (llama.cpp does not support
    # batch generation like transformers)
    for i, (prompt, sid) in tqdm(
        enumerate(zip(prompts, sample_ids)),
        total=len(prompts),
        desc="Generating task descriptions",
    ):
        output = model(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
        )

        # llama.cpp returns a dict with choices list
        response = output["choices"][0]["text"]

        task_descriptions.append(response)
        model_output_rows.append((sid, response))

        # Save intermediate model outputs to avoid loss and enable
        # resuming the experiment at roughly the last sample
        if (i + 1) % SAVE_EVERY == 0 or i == len(prompts) - 1:
            batch_sids, batch_outputs = zip(*model_output_rows[-SAVE_EVERY:])
            intermediate_df = pd.DataFrame(
                {
                    "sample_id": batch_sids,
                    "model_output": batch_outputs,
                }
            )

            intermediate_df.to_csv(
                MODEL_OUTPUT_PATH,
                mode="a",
                header=not MODEL_OUTPUT_PATH.exists(),
                index=False,
            )

    # Generate embeddings in batch
    task_description_embeddings = embedding_model.encode(
        task_descriptions, batch_size=64  # depends on memory
    )

    df["task_description"] = task_descriptions
    df["task_description_embedding"] = list(task_description_embeddings)

    train_df, eval_df = train_test_split(
        df.copy(), test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE
    )

    # Convert promp embeddings to arrays
    train_embeddings = np.stack(train_df["task_description_embedding"])
    eval_embeddings = np.stack(eval_df["task_description_embedding"])

    # Normalize embeddings for better nearest neighbors performance
    train_embeddings = normalize(train_embeddings)
    eval_embeddings = normalize(eval_embeddings)

    # Use response embeddings as a baseline
    train_prompt_embeddings = np.stack(train_df[PROMPT_EMBEDDING_COL])
    eval_prompt_embeddings = np.stack(eval_df[PROMPT_EMBEDDING_COL])
    train_prompt_embeddings = normalize(train_prompt_embeddings)
    eval_prompt_embeddings = normalize(eval_prompt_embeddings)

    for k in K_VALUES:

        print(f"Running evaluation for {k=}")
        start = time.time()

        predictions = predict_top_k(
            eval_embeddings,
            train_embeddings,
            train_df,
            k,
        )

        accuracy, correct, total = evaluate_predictions(eval_df, predictions)

        elapsed = time.time() - start

        results_rows.append(
            {
                "embedding_type": "task_description_embedding",
                "k": k,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "time_sec": elapsed,
            }
        )

        # Now do the same thing with the response embeddings for a basline
        start = time.time()

        predictions = predict_top_k(
            eval_prompt_embeddings,
            train_prompt_embeddings,
            train_df,
            k,
        )

        accuracy, correct, total = evaluate_predictions(eval_df, predictions)

        elapsed = time.time() - start

        results_rows.append(
            {
                "embedding_type": "prompt_embedding",
                "k": k,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "time_sec": elapsed,
            }
        )

    # Ensures resources are released cleanly, can cause errors otherwise
    model.close()

    # Save results to CSV for easy viewing
    results_df = pd.DataFrame(results_rows)
    print(results_rows)
    results_df.to_csv(RESULTS_CSV_PATH, index=False)


def _run_experiment() -> None:
    if RESULTS_CSV_PATH.exists():
        print(f"Experiment 9 results already exist at: {RESULTS_CSV_PATH}")
    else:
        _evaluate_llamacpp_model()


if __name__ == "__main__":
    _run_experiment()

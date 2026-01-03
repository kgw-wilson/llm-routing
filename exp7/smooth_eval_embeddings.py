from pathlib import Path
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from data_processing.constants import (
    RESULTS_CSV_FILENAME,
    RESPONSE_EMBEDDING_SUFFIX,
    MODEL_PERFORMANCE_COLS,
    PROMPT_EMBEDDING_COL,
    FILTERED_WITH_EMBEDDINGS_PATH,
)
from utils import (
    TEST_SPLIT_SIZE,
    RANDOM_STATE,
    K_VALUES,
    predict_top_k,
    evaluate_predictions,
)

# Directory for exp7 - (where to save results)
EXP7_PATH = Path(__file__).parent.parent / "exp7"
RESULTS_CSV_PATH = EXP7_PATH / RESULTS_CSV_FILENAME

SMOOTHED_EMBEDDING_SUFFIX = "|smoothed"


def _run_experiment() -> None:
    """
    Run kNN evaluation using prompt-relative neighborhood embeddings:
    Each promtp is represented by the mean of its top-k nearest responses
    in the training set.

    1.	Find similar prompts in the training set
	2.	Look at the responses that those prompts elicited
	3.	Average those responses → this gives you a prototype response embedding
	4.	Use that embedding when predict the model
    """

    df = pd.read_pickle(FILTERED_WITH_EMBEDDINGS_PATH)

    results_rows = []

    train_df, eval_df = train_test_split(
        df, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE
    )

    # 1. Compute prompt similarity
    prompt_train = normalize(np.stack(train_df[PROMPT_EMBEDDING_COL]))
    prompt_eval = normalize(np.stack(eval_df[PROMPT_EMBEDDING_COL]))
    sims = cosine_similarity(prompt_eval, prompt_train)

    # Set an initial value for k_smooth - determines how many similar
    # prompts to consider in the training data
    k_smooth = 5

    for model_name in MODEL_PERFORMANCE_COLS:
        for k in K_VALUES:


            print(f"Running evaluation for {model_name} with {k=}")

            # 2. Use prompt neighbors to average RESPONSE embeddings
            response_train = normalize(np.stack(train_df[model_name+RESPONSE_EMBEDDING_SUFFIX]))

            smoothed_eval_responses = []
            for i in range(len(sims)):
                idx = np.argsort(sims[i])[::-1][:k_smooth]
                smoothed_eval_responses.append(response_train[idx].mean(axis=0))

            smoothed_eval_responses = np.stack(smoothed_eval_responses)
            
            start = time.time()

            predictions = predict_top_k(
                smoothed_eval_responses,
                response_train,
                train_df,
                k,
            )

            accuracy, correct, total = evaluate_predictions(eval_df, predictions)

            elapsed = time.time() - start

            results_rows.append(
                {
                    "k": k,
                    "model_name": model_name,
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total,
                    "time_sec": elapsed,
                }
            )

    results_df = pd.DataFrame(results_rows)
    print(f"Saving experiment results to: {RESULTS_CSV_PATH}")
    results_df.to_csv(RESULTS_CSV_PATH, index=False)


if __name__ == "__main__":
    if RESULTS_CSV_PATH.exists():
        print(f"Experiment 7 results already exist at: {RESULTS_CSV_PATH}")
    else:
        _run_experiment()

from pathlib import Path
import time
import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
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


# Directory for exp6 - (where to save results)
EXP6_PATH = Path(__file__).parent.parent / "exp6"
RESULTS_CSV_PATH = EXP6_PATH / RESULTS_CSV_FILENAME

ANGLE_SPACE_SUFFIX = "|angle_space"


def _run_experiment() -> None:
    """
    Run a kNN style evaluation using angle space constructed from response and prompt embeddings
    """

    df = pd.read_pickle(FILTERED_WITH_EMBEDDINGS_PATH)

    print("Generating angle space for all models...")

    for model_name in MODEL_PERFORMANCE_COLS:
        prompt_col = PROMPT_EMBEDDING_COL
        response_col = f"{model_name}{RESPONSE_EMBEDDING_SUFFIX}"

        def angle_features(row):
            p = row[prompt_col]
            r = row[response_col]
            cosine_sim = np.dot(p, r) / (norm(p) * norm(r) + 1e-10)
            r_norm = norm(r)
            diff_norm = norm(r - p)
            return np.array([cosine_sim, r_norm, diff_norm], dtype=np.float32)

        df[model_name + ANGLE_SPACE_SUFFIX] = df.apply(angle_features, axis=1)

    train_df, eval_df = train_test_split(
        df, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE
    )

    # List for saving experiment results will be turned into DataFrame later
    results_rows = []

    for model_name in MODEL_PERFORMANCE_COLS:

        angle_space_col = model_name + ANGLE_SPACE_SUFFIX

        # Convert angle spaces to arrays
        train_embeddings = np.stack(train_df[angle_space_col])
        eval_embeddings = np.stack(eval_df[angle_space_col])

        # Normalize embeddings for better nearest neighbors performance
        # train_embeddings = normalize(train_embeddings)
        # eval_embeddings = normalize(eval_embeddings)

        for k in K_VALUES:

            print(f"Running evaluation for {model_name} with {k=}")
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
                    "model_name": model_name,
                    "k": k,
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total,
                    "time_sec": elapsed,
                }
            )

    print(f"Saving experiment results to: {RESULTS_CSV_PATH}")
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(RESULTS_CSV_PATH, index=False)


if __name__ == "__main__":
    if RESULTS_CSV_PATH.exists():
        print(f"Experiment 6 results already exist at: {RESULTS_CSV_PATH}")
    else:
        _run_experiment()

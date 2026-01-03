from pathlib import Path
import time
import pandas as pd
import numpy as np
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


# Directory for exp3 - (where to save results)
EXP3_PATH = Path(__file__).parent.parent / "exp3"
RESULTS_CSV_PATH = EXP3_PATH / RESULTS_CSV_FILENAME

CONCATENATED_EMBEDDING_SUFFIX = "|concatenated_embedding"


def _run_experiment() -> None:
    """
    Run a kNN style evaluation using concatenated prompt and response embeddings

    This could be vectorized for better efficiency, but I like
    how the loop over the model names makes it clear how all models
    are being evaluated (which is different from exp1).
    """

    df = pd.read_pickle(FILTERED_WITH_EMBEDDINGS_PATH)

    print("Concatenating prompt and response embeddings for all models...")

    for model_name in MODEL_PERFORMANCE_COLS:

        df[model_name + CONCATENATED_EMBEDDING_SUFFIX] = df.apply(
            lambda row: np.concatenate(
                [
                    row[PROMPT_EMBEDDING_COL],
                    row[f"{model_name}{RESPONSE_EMBEDDING_SUFFIX}"],
                ]
            ),
            axis=1,
        )

    train_df, eval_df = train_test_split(
        df, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE
    )

    # List for saving experiment results will be turned into DataFrame later
    results_rows = []

    for model_name in MODEL_PERFORMANCE_COLS:

        concatenated_embedding_col = model_name + CONCATENATED_EMBEDDING_SUFFIX

        # Convert concatenated embeddings to arrays
        train_embeddings = np.stack(train_df[concatenated_embedding_col])
        eval_embeddings = np.stack(eval_df[concatenated_embedding_col])

        # Normalize embeddings for better nearest neighbors performance
        train_embeddings = normalize(train_embeddings)
        eval_embeddings = normalize(eval_embeddings)

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
        print(f"Experiment 3 results already exist at: {RESULTS_CSV_PATH}")
    else:
        _run_experiment()

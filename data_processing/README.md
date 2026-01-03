# Data Processing

This directory contains scripts that download and process data.

To start, run

```shell
huggingface-cli login
```

and provide HuggingFace with your access token. Without this, you won't be able to download the RouterBench data. 

Then, you can run the files in this order:

1. download_filter_rb_data.py > Downloads RouterBench 0-shot data from HuggingFace into the project and filters out columns which aren't usable.
2. add_pr_embeddings.py > Uses the saved data from step 1 to generate embeddings (512 x 512 arrays describing the content of each paragraph) for both prompts and responses.

Then you're good to go with the experiments.

## Downloading Data Rationale

We chose to consider the 0-shot data only rather than any of the other data because it meant less data (prompts no longer contain any examples before the actual task), and because the embedding generated from a prompt would be less noisy (no extra information other than the actual task).

Warning: requires about 1 GB of space
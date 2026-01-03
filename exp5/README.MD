# Experiment 5: Weighted Difference

Experiment 4 applied kNN on the simple difference between response and prompt embeddings, but this yielded only minimal performance gains over using prompt or response embeddings alone.

In Experiment 5, we tested whether a weighted difference might improve results. Weights alpha and beta were chosen from \{0.1, 0.3, 0.5, 0.7, 0.9\}, with alpha + beta = 1.


## Results

Weighted differences showed negligible improvement, confirming that weighted differences didn't show a lot of promise or unweighted response embeddings are sufficient.

It was planned to vary this experiment by weighting by prompt/response length, but benefits seem unlikely.


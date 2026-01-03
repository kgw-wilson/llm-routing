# Experiment 5: Weighted Difference

In Experiment 4, we applied kNN routing using the simple vector difference between response and prompt embeddings. This approach produced only marginal improvements over using prompt or response embeddings alone.

Experiment 5 investigates whether introducing explicit weighting between the prompt and response embeddings can improve performance. Instead of using an unweighted difference, we compute a weighted difference of the form:

$$
\alpha \cdot r - \beta \cdot p \quad \text{where} \quad \alpha + \beta = 1
$$

The weights alpha and beta were selected from the set {0.1, 0.3, 0.5, 0.7, 0.9}, allowing us to explore a small range of prompt- vs. response-dominant representations.

## Results

Across all tested weight combinations, the weighted difference embeddings showed negligible improvement over the unweighted difference and over using response embeddings alone. In practice, performance remained nearly identical to Experiment 4.

These results suggest that:

• Explicitly tuning linear weights between prompt and response embeddings in this simple manner does not meaningfully enhance routing accuracy.

• Either the unweighted difference already captures most of the useful signal, or the response embedding alone is sufficient for this task.

## Takeaway

Weighted linear combinations of prompt and response embeddings do not appear to be a promising direction for improving kNN-based routing. We considered further variants (such as weighting based on prompt or response length) but the observed results make it unlikely that such extensions would yield substantial gains.
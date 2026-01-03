# Experiment 6: kNN on Prompt Response Angle Space

This experiment explores whether low-dimensional geometric features derived from prompt and response embeddings are sufficient for effective model routing.

Instead of performing kNN directly in the high-dimensional embedding spaces (as in Experiment 1), we project each prompt–response pair into a compact 3D “angle space” defined by:

$$
\text{features} =
\begin{bmatrix}
\cos(p, r) \\
\|r\| \\
\|r - p\|
\end{bmatrix}
$$

Where:

• p is the prompt embedding

• r is the model’s response embedding

• cos(p, r) captures semantic alignment between prompt and response

• |r| reflects response magnitude

• |r - p| measures deviation from the prompt

These features are intentionally not normalized, as normalization would remove magnitude-related information that may be useful for routing.

## Results and Interpretation

This approach resulted in an average ~1.2% accuracy drop compared to Experiment 1 (kNN on full prompt embeddings).

This outcome suggests that while prompt–response geometric relationships contain meaningful signal, high-dimensional embedding structure retains important information that is lost when compressing into this low-dimensional summary features.

In other words, the rich semantic detail encoded across the full embedding vectors appears to be beneficial for accurate routing decisions, and cannot be fully captured by simple geometric descriptors alone.

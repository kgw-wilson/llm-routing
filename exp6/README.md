# Experiment 6: kNN on Prompt Response Angle Space

$$
\text{features} =
\begin{bmatrix}
\text{cosine}(p, r) \\
\|r\| \\
\|r - p\|
\end{bmatrix}
$$

This resulted in average 1.2% accuracy drop below experiment 1, meaning information incoded in the high dimensinoal prompt and response embeddings is good for classification.
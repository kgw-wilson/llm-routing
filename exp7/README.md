# Experiment 7: Relative Neighborhood Embeddings

Experiment 6 suggested that embedding dimensionality plays a significant role in accurate model selection. This experiment explores whether performance can be improved without changing dimensionality, by instead smoothing embeddings using local neighborhood information.

To construct relative neighborhood embeddings, we smooth response embeddings using similarity computed in prompt embedding space.

For each evaluation sample, we first identify its k nearest neighbors in the training set based on prompt embedding similarity. Rather than using the evaluation sample’s own response embedding for routing, we compute the mean of the corresponding response embeddings from those nearest neighbors.

1. All training response embeddings are normalized.

2. For each evaluation sample, the top-k most similar training prompts are selected.

3. The response embeddings associated with those prompts are averaged to form a smoothed representation.

4. This smoothed response embedding is then used as input to the kNN routing step.

This approach decouples similarity measurement (prompt space) from routing representation (response space), allowing the router to leverage neighborhood-level task structure while preserving the dimensionality of the original embeddings.
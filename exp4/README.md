# Experiment 4 Difference of Prompt and REsponse embeeddings

In this experiment, we represent each sample using the vector difference between the response embedding and the corresponding prompt embedding:

$$
\text{task\_vector} = \text{response\_embedding} - \text{prompt\_embedding}
$$

## Intuition

This representation aims to capture what information the model adds beyond the prompt itself. Intuitively, the difference vector isolates the transformation induced by the model when generating a response, potentially encoding task-specific reasoning or completion behavior rather than raw prompt content.

## Results

Performance using the difference vector was comparable to Experiment 3. The mean accuracy across all models was approximately 0.5% higher than Experiment 1 (prompt embeddings) and 1% higher than Experiment 2 (response embeddings).

While this suggests that incorporating both prompt and response information can provide a modest benefit, the improvement is small. Overall, simple vector differencing does not appear to capture substantially more useful routing signal than higher-dimensional representations that do not modify the embedding structure.
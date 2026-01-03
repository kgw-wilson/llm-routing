# Concatenation of Prompt and Response Embeddings

Given that kNN performed reasonably well with both the prompt and responses, start trying other things like combinations of them both. Simpleset is concatenation.

## Results

1. Concatenation impact depends on model

Because the experiment did not result in a clear increase or decrease across all values of k when compared to experiment 2, it seems that the way embeddings encode task info varies by model.

2.	Concatenation often improves low- and mid-k performance:

Particularly for k=5 and k=10, models like WizardLM-13B, claude-v1, meta/llama-2-70b-chat see substantial gains. This suggests that concatenating prompt + response embeddings gives more task-relevant information when considering smaller neighborhoods.

3.	High-k performance is mixed:

For k=40, gains are smaller and sometimes slightly negative. Maybe response embeddings alone already capture task similarity well for large neighborhoods, so concatenating prompt embeddings just adds some noise.
    
4.	Tradeoff with runtime:

Concatenated embeddings slightly increase runtime (~6.7–7.8s vs ~5.7–6.3s), which is expected because embeddings are twice as long.
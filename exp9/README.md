# kNN on Prompt Desription Embeddings

This experiment investigates whether task description embeddings can replace full prompt embeddings for kNN-based model routing.

Instead of embedding the entire prompt, we generate a short natural-language description of the task represented by the prompt and use that embedding for kNN evaluation to select the best model.

## Motivation

Prompt embeddings are effective for routing, but they are often long and verbose. If a shorter task description captures the same semantic signal, it could offer a simpler and potentially more efficient routing representation.

## Findings

We explored many ways of generating task descriptions, including:

• Different task-description prompts

• Single-sentence descriptions enforced via stop tokens

• Few-shot task description examples

• Variations in task template details (adding/removing instructions, enforcing more/less strict output structure)

Across these variations, most task description embeddings produced kNN clustering performance comparable to prompt embeddings, matching the results reported in results.csv.

Interestingly, task description embeddings are significantly shorter on average while achieving similar accuracy. However, this does not materially improve routing performance, suggesting that most of the useful signal already exists in the full prompt rather than in an abstracted task description.

Notably:

• Enforcing single-sentence outputs had no measurable effect

• Few-shot task description examples tended to reduce quality, as the model often repeated the examples

• Beyond providing few-shot examples, changes to the task template details had minimal impact

## Conclusion

Task description embeddings can perform about as well as prompt embeddings for kNN routing, but they do not offer a clear advantage beyond reduced length. For practical routing purposes, prompt embeddings remain sufficient and simpler to reason about.

## Reproducibility Note

Running this experiment requires generating task descriptions for the full RouterBench dataset. On a MacBook M1, this took over 2 days.

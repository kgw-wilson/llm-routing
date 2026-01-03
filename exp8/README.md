# Experiment 8: Package Comparison for Quantized LLMs on Macbook M1

This experiment compares Hugging Face Transformers against llama.cpp for performing inference using an LLM (mistral 7b instruct).

Initially, I attempted to use bitsandbytes to load a quantized model on my MacBook. However, on Apple Silicon:

• BitsAndBytes does not support MPS (GPU acceleration), since it relies on CUDA, which is NVIDIA-only.

• Running Transformers on CPU without quantization is very slow and memory intenseive.

Forcing 4-bit quantization on CPU worked, but inference took ~953 seconds for a single prompt. I could switch to Colab to use their free GPUs, but this adds complexity with timesouts and mounting files and such so I'd rather keep everything local.

## Using llama.cpp

I switched to llama.cpp, which in their words treats Apple Silicon as a “first-class citizen” and provides much faster generation times over CPU only. To use it:

1.	Install llama.cpp:

```shell
brew install llama.cpp
```

2. Install the chosen model from HuggingFace:

```shell
llama-cli \
  -hf MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF \
  --hf-file Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
```

Everything has downloaded successfully if you see a big llama.cpp logo and can type `/exit` to leave their interactive shell.

3. Supply the path to the saved .gguf file when running the experiment (for instance):

```shell
python -m exp8.package_comparison \
    --llamacpp_model_path="/users/<your_username>/library/caches/llama.cpp/MaziyarPanahi_Mistral-7B-Instruct-v0.3-GGUF_Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
```

Notes:

• Q4_K_M → 4-bit integer quantization (Q4-style) with some optimizations to reduce error vs standard 4-bit quantization.

• M → Medium quality / medium memory footprint.

• llama.cpp automatically detects which runtime to use, so it should work fine if anyone else tries to replicate this experiment. 

## Results

1. I could not get Transformers + bitsandbytes to run on MPS (Apple GPU) due to PyTorch and bitsandbytes CUDA limitations. ([StackOverflow reference](https://stackoverflow.com/questions/76924239/accelerate-and-bitsandbytes-is-needed-to-install-but-i-did))

2. 4-bit CPU inference with Transformers worked, but was very slow (~953s per prompt).

3. LLaMA.cpp 4-bit inference on M1 completed in ~6 seconds, making it far more practical.

So for Apple Silicon, llama.cpp is what future experiments will use.

Other notes: model load time from the transformers package is much longer (~126s vs 2.5s) which adds to the reason to use llama.cpp.

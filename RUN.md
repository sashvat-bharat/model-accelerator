# Command Reference for Model Accelerator

## 🛠 Parameters & Flags Description

### 🚀 Performance & Hardware Flags
These flags control how the model interacts with your CPU and GPU.
* `--gpu-layers`: Controls GPU offloading. 
    * `max`: (Recommended) Automatically detects the maximum number of layers that fit in VRAM and applies a "Compute Margin" to prevent OOM crashes during generation.
    * `<int>`: Forces a specific number of layers onto the GPU.
* `--n-ctx`: Sets the total context window size (KV Cache). Defaults to 4096 or the model's native limit.
* `--n-batch`: The number of tokens to process in a single batch during prefill. Default is 512.
* `--n-threads`: Number of CPU threads used for generation eval. Defaults to your core count (max 8).

### 🧊 Sampling & Logic Flags
These flags control the "creativity" and behavior of the model.
* `--chat`: **Crucial Flag.** Tells the engine to extract the native Jinja2 chat template from the GGUF metadata and format the prompt correctly (e.g., Llama-3, ChatML, Mistral).
* `--think`: Enables thinking mode for models that support it. The model will output its reasoning in `<think>` tags before the final answer. Silently ignored for models without thinking capability.
* `--temp`: Temperature for sampling (0.0 to 2.0). Higher is more creative, 0 is deterministic.
* `--max-tokens`: The maximum number of tokens to generate. Use `auto` to dynamically use remaining context.
* `--top-p` / `--top-k`: Advanced sampling filters to control token diversity.

---

## 📖 Feature Commands

### 1. Load Model
Downloads a model from Hugging Face or a URL and registers it with a short alias.
* **Flags:**
    * `--alias`: (Required) The name you will use to call the model.
    * `--file`: (Optional) Specify a specific `.gguf` file if the HF repo contains many.

```bash
# Load from Hugging Face (Auto-selects best quantization)
ma load unsloth/Qwen3.5-9B-GGUF --alias qwen3_5-9b

# Load a specific file from a repo
ma load unsloth/Qwen3.5-9B-GGUF --alias qwen3_5-9b --file Qwen3.5-9B-Q5_K_M.gguf
```

### 2. Run Model (Single Prompt)
Generates a response for a single input. Supports streaming by default.
* **Flags:**
    * `--prompt`: (Required) The text input.
    * `--no-stream`: Disables real-time token printing.

```bash
ma run qwen3_5-9b --prompt "Explain Entropy in 2 sentences." --chat --gpu-layers max
```

### 3. Interactive Chat
Starts a persistent REPL session with conversational memory.
* **Flags:**
    * `--system`: Sets the behavior of the assistant.

```bash
ma chat qwen3_5-9b --system "You are a concise Linux expert." --gpu-layers max --n-ctx 8192
```

### 4. Parallel Batch Processing
Processes multiple prompts from a text file simultaneously using low-level C API batching.
* **Flags:**
    * `--file`: (Required) Path to a `.txt` file with one prompt per line.
    * `--output`: Saves results to a `.json` file.
    * `--parallel`: Number of prompts to process in a single parallel sub-batch.

```bash
ma batch qwen3_5-9b --file prompts.txt --output results.json --parallel 25 --chat --gpu-layers max
```

### 5. Model Configuration
Views or saves default parameters for a specific model so you don't have to type them every time.
* **Flags:**
    * `--max-output` / `--max-input`: Set to `auto` or a specific integer.
    * Sets defaults for `temp`, `top-p`, `top_k`, and `n-ctx`.

```bash
# Save Q4_K_M settings for qwen
ma config qwen3_5-9b --n-ctx 4096 --max-output 512 --max-input auto --temp 0.7 --top-p 1.0 --top-k 0
```

### 6. Throughput Benchmark
Tests the raw performance of your hardware with the selected model.
* **Flags:**
    * `--runs`: How many times to repeat the test for averaging.

```bash
ma bench qwen3_5-9b --prompt "Write a long essay on AI." --runs 5 --gpu-layers max
```

### 7. List Models
Displays all models currently registered in your system, their sizes, and their sources.

```bash
ma list
```
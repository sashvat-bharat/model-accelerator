# 🚀 model-accelerator

> **The fastest, most efficient library for running GGUF models with maximum throughput and zero-config hardware optimization.**

`model-accelerator` isn't just another wrapper. It’s a high-performance inference engine designed to squeeze every drop of power out of your hardware. By talking directly to GGUF metadata, it configures itself perfectly for every model you throw at it.

---

## ⚡ Quick Start

Get up and running in seconds. Optimized for `uv` and high-performance inference.

```bash
# 1. Setup Environment
uv sync && source .venv/bin/activate

# Optional: Enable GPU/CUDA support
# Using Release build type is the standard way to get -O3 optimizations automatically
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=on" uv pip install llama-cpp-python --reinstall --no-binary llama-cpp-python



# 2. Load & Register a Model
# Pulls directly from HF, auto-selecting the best quantization (Q4_K_M/Q5_K_M).
ma load unsloth/Qwen3.5-9B-GGUF --alias qwen3_5-9b

# 3. Configure (Optional: Set your hardware's "sweet spot")
ma config qwen3_5-9b --n-ctx 8192 --max-output auto --temp 0.7

# 4. Interact!
ma chat qwen3_5-9b --gpu-layers max
```

---

## 🛠️ Command Reference

For a deep dive into every possible flag and advanced usage examples, check out our **[Full Command Guide (RUN.md) ↗](./RUN.md)**.

| Command | Usage Example | Description |
|:---|:---|:---|
| **load** | `load <src> --alias <name>` | Download from HF/URL or register a local path. |
| **chat** | `chat <alias>` | Interactive REPL. Uses native GGUF Jinja2 templates. |
| **run** | `run <alias> --prompt "..."` | Single-prompt generation with real-time streaming. |
| **batch** | `batch <alias> --file <f>` | High-throughput C-level parallel processing. |
| **config** | `config <alias> [flags]` | View/Set persistent sampling & resource limits. |
| **bench** | `bench <alias>` | Benchmark throughput (tokens/sec) for peak speed. |
| **list** | `list` | Show all registered models and disk sizes. |

---

## ✨ The Engine Experience

We built this engine for developers who want precision and feedback. No more "guessing" if your settings are working:

- **Bulletproof Self-Healing:** Using `--gpu-layers max`, the engine reads the model's exact internal layer count. If you run out of VRAM, it automatically steps down layer-by-layer and applies a **Compute Margin** to ensure you have enough memory left for the actual math—preventing those annoying CUDA crashes.
- **Native GGUF Intelligence:** It extracts Chat Templates, EOG (End-of-Generation) tokens, and Context Limits directly from the model file. It knows exactly how to talk to your model without you needing to tell it.
- **Rich Terminal UI:** Beautiful, clean output using `rich`. You get real-time feedback on parameters, token speed, and wall-clock time for every single generation.
- **C-Level Parallelism:** The `batch` command bypasses Python's bottlenecks to run multiple sequences at once using native `llama.cpp` bindings for massive throughput.

---

## 🏎️ Performance Tuning

You can add these flags to any command or save them permanently via `config`.

### 🎮 Resource Allocation
*   `--gpu-layers max`: **(Recommended)** Automatically finds the absolute maximum layers your GPU can handle with a safety buffer.
*   `--n-ctx <N>`: Set your context window (e.g., `8192` or `32768`).
*   `--n-threads <N>`: CPU threads. We default to your physical core count (max 8) for the best balance.

### 🎲 Sampling & Logic
*   `--chat`: Always use this for Instruct/Chat models! It applies the model's specific "brain" format to your prompt.
*   `--temp <F>`: Controls randomness (0.0 = deterministic, 0.7 = standard, 1.2 = creative).
*   `--think`: Enables thinking mode for models that support it. The model will output its reasoning in `<think>` tags before the final answer. Silently ignored for models without thinking capability.

---

## 🏗️ Architecture & Data

### Project Structure
```text
model-accelerator/
├── engine.py          # The Brain: Dataclass-driven, handles GGUF metadata & self-healing
├── cli.py             # The Interface: Rich-powered CLI for interaction
├── pyproject.toml     # Dependencies (rich, llama-cpp-python, jinja2)
├── models.json        # The Registry: Maps aliases to your local storage
└── models/            # Storage: Houses your GGUF files and per-model params.json
```

---

## 💎 Pro Tips

- **Stop Typing Flags:** Use `ma config <alias> --gpu-layers max --n-ctx 4096`. These settings are saved to `params.json` and inherited automatically by `run`, `chat`, and `batch`.
- **Environment Variables:** You can set `export GGUF_MODELS_DIR=/path/to/large/drive` to keep your models on secondary storage.
- **Bulk Processing:** Need to categorize 1,000 strings? `ma batch my-model --file prompts.txt --parallel 20` will process them in parallel chunks.

---
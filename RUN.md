# 🏎️ Model Accelerator: Clean Command Guide

This guide provides everything you need to run, tune, and optimize your LLMs with maximum efficiency. For installation details, see the [Main README](./README.md).

---

## ⚡ Quick Start
Get up and running in three commands:
```bash
# 1. Download & Register (Specific File Selection)
python cli.py load unsloth/gpt-oss-20b-GGUF --alias gptoss_20b --file gpt-oss-20b-Q4_K_M.gguf

# 2. Run it once to check if it works
python cli.py run gptoss_20b --prompt "Hello, how are you?" --gpu-layers max --chat

# 3. Configure (Optional: Set persistent context/limits)
python cli.py config gptoss_20b --n-ctx 8192 --max-output auto

# 4. Chat!
python cli.py chat gptoss_20b
```

---

## 🛠️ Command Reference

### 📦 Model Management
| Command | Usage | Description |
|:---|:---|:---|
| **load** | `python cli.py load <source> --alias <name> [--file <filename>]` | Download from HF, URL, or load local path. |
| **list** | `python cli.py list` | Show all registered models and their disk sizes. |

### 💬 Inference Modes
| Command | Usage | Description |
|:---|:---|:---|
| **chat** | `python cli.py chat <alias>` | Interactive REPL. Auto-detects chat templates. |
| **run** | `python cli.py run <alias> --prompt "..."` | Generate text for a single prompt. |
| **batch**| `python cli.py batch <alias> --file prompts.txt`| High-throughput parallel processing of multiple prompts. |

### ⚙️ Optimization & Tuning
| Command | Usage | Description |
|:---|:---|:---|
| **config**| `python cli.py config <alias>` | View/Set persistent sampling & resource limits. |
| **bench** | `python cli.py bench <alias>` | Benchmark throughput (tokens/sec) to find your peak speed. |

---

## 🚀 Performance Tuning
Add these flags to **any** inference command (`run`, `chat`, `batch`, `bench`) to override defaults.

### 🎮 GPU Offloading
*   `--gpu-layers max`: **Recommended.** Automatically calculates the maximum layers your VRAM can fit using granular 1-layer backtracking.
*   `--gpu-layers -1`: Attempt to offload all layers to GPU.
*   `--gpu-layers 0`: CPU-only mode.

### 🧠 Resource Allocation
*   `--n-ctx <N>`: Context window size (e.g., `8192`). Larger windows use more VRAM.
*   `--n-batch <N>`: Number of tokens processed per batch. `512` is standard; `2048` may increase speed on high-end GPUs.
*   `--n-threads <N>`: Number of CPU threads. Best results usually around `8`.

### 🎲 Sampling Defaults
*   `--temp <F>`: Randomness (0.1 = strict, 0.7 = creative).
*   `--chat`: Apply model-specific chat headers to your raw prompt.

---

## 💎 Pro Tips

### 💾 Persistent Settings
Stop typing flags every time! Use `config` to save your hardware's "sweet spot" settings.
```bash
python cli.py config gptoss_20b --n-ctx 8192 --temp 0.8 --max-output 2048
```
*Settings are saved to `models/<alias>/params.json` and inherited by all commands.*

### 📂 Batch Export
Process thousands of prompts and save results directly to a JSON file:
```bash
python cli.py batch gptoss_20b --file prompts.txt --parallel 10 --output results.json --chat
```

### 🌍 Environment Variables
Instead of CLI flags, you can set global environment variables for your session:
```bash
export LLM_GPU_LAYERS=max
export LLM_N_BATCH=1024
```

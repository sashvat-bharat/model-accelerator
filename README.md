# 🚀 model-accelerator

> **The fastest, most efficient library for running AI models with maximum throughput at any scale.**

---

## ⚡ Quick Start

Get up and running in seconds. Optimized for `uv` and high-performance inference.

```bash
# 1. Setup Environment
uv venv && source .venv/bin/activate
uv pip install -e .

# Optional: Enable GPU/CUDA support
CMAKE_ARGS="-O3 -DGGML_CUDA=on" uv pip install llama-cpp-python --reinstall --no-binary llama-cpp-python

# 2. Load & Register a Model
# Supports HF repos, direct URLs, or local files.
python cli.py load unsloth/gpt-oss-20b-GGUF --alias gptoss_20b

# 3. Configure (Optional: Set persistent hardware limits)
python cli.py config gptoss_20b --n-ctx 8192 --max-output auto

# 4. Interact!
python cli.py chat gptoss_20b
```

---

## 🛠️ Command Reference

| Command | Usage Example | Description |
|:---|:---|:---|
| **load** | `load <src> --alias <name> [--file <f>]` | Download from HF/URL or register a local path. |
| **chat** | `chat <alias>` | Interactive REPL. Auto-detects chat templates. |
| **run** | `run <alias> --prompt "..." [--chat]` | Single-prompt generation (streaming). |
| **batch** | `batch <alias> --file <f> [--output <j>]` | Parallel processing of multiple prompts. |
| **config** | `config <alias> [flags]` | View/Set persistent sampling & resource limits. |
| **bench** | `bench <alias>` | Benchmark throughput (tokens/sec) for peak speed. |
| **list** | `list` | Show all registered models and disk sizes. |

---

## ✨ Professional Experience

The engine is built for feedback and precision, providing a heavy-duty terminal interface:
- **GENERATION Block:** Real-time feedback on resolved parameters (Temp, Max Tokens, Mode).
- **Dynamic Speed-Line:** High-accuracy `tokens/sec` and wall-clock tracking for every generation.
- **Granular VRAM Backtracking:** Automatically finds the absolute maximum layers your GPU can handle (1-layer precision) using `--gpu-layers max`.
- **Segmented Model Support:** Native handling for multi-channel models like **GPT-OSS** (visualizes thinking/analysis phases).

---

## 🏎️ Performance Tuning

Tuning hardware parameters is key to maximizing speed. Add these flags to any command or set them via `config`.

### 🎮 Resource Allocation
*   `--gpu-layers <N>`: `-1` (all), `0` (CPU), or **`max` (Auto-calculate best VRAM fit)**.
*   `--n-ctx <N>`: Context window size (e.g., `8192`).
*   `--n-batch <N>`: Token batch size (e.g., `512` standard, `2048` for high-end GPUs).
*   `--n-threads <N>`: CPU threads. Best results usually around `8`.

### 🎲 Sampling & Logic
*   `--temp <F>`: Randomness (0.1 = strict/bench, 0.7 = creative).
*   `--chat`: Apply model-specific chat headers to raw prompts.
*   `--parallel <N>`: (Batch only) Number of concurrent sequences.

---

## 🏗️ Architecture & Data

### Project Structure
```text
model-accelerator/
├── engine.py          # The Brain: Handles zero-reload overlay & smart GPU detection
├── cli.py             # The Interface: Handles acquisition and user interaction
├── pyproject.toml     # Dependencies
├── models.json        # The Registry: Tracks aliased models (gitignored)
└── models/            # Storage: Where .gguf files and params.json live (gitignored)
```

### Registry Example (`models.json`)
The registry maps aliases to paths and sources:
```json
{
  "gptoss_20b": {
    "path": "./models/gptoss_20b.gguf",
    "source": "unsloth/gpt-oss-20b-GGUF"
  }
}
```

---

## 💎 Pro Tips

- **Persistent Settings:** Stop typing flags! Use `config` to save your hardware's "sweet spot" to `models/<alias>/params.json`. They are inherited by all commands.
- **Bulk Export:** Process thousands of prompts: `python cli.py batch gptoss --file q.txt --output results.json --chat`.
- **Global Env Vars:** Use `export LLM_GPU_LAYERS=max` or `LLM_N_BATCH=1024` for session-wide defaults.
- **Mmap support:** Models use memory-mapping for instant startup.

---

## 🔮 Future Work

- [ ] **Continuous Batching:** Scheduler for simultaneous sequence execution.
- [ ] **OpenAI-Compatible Server:** HTTP endpoint via `llama_cpp.server`.
- [ ] **LoRA Support:** Native adapter loading.


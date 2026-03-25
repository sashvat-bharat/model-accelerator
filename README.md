# 🚀 model-accelerator

> **The fastest, most efficient library for running AI models with maximum throughput at any scale.**
---

## ⚡ Quick Start

```bash
# 1. Setup the environment (Using uv)
uv venv
source .venv/bin/activate
uv pip install -e .

# Optional: For strict GPU support (CUDA)
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --reinstall --no-binary llama-cpp-python

# 2. Register/Download a model 
python cli.py load unsloth/gpt-oss-20b-GGUF --alias gptoss_20b

# 3. Configure it (Optional - set persistent context/limits)
python cli.py config gptoss_20b --n-ctx 8192 --max-output auto

# 4. Chat with it!
python cli.py chat gptoss_20b
```

---

## 🛠️ CLI Commands Overview

| Command | Usage | Description |
|---|---|---|
| **load** | `python cli.py load <source> --alias <name>` | Download & register a model (HF repo, URL, or local path) |
| **chat** | `python cli.py chat <alias>` | Start an interactive REPL chat (automatically uses model's chat template) |
| **run** | `python cli.py run <alias> --prompt "..." [--chat]` | Generate text for a single prompt (optional chat template) |
| **batch** | `python cli.py batch <alias> --file <file> [--chat]` | Run inference over a text file of prompts (optional chat template) |
| **bench** | `python cli.py bench <alias>` | Run throughput benchmarks to test your token speed |
| **list** | `python cli.py list` | Show all registered models in your local registry |
| **config** | `python cli.py config <alias>` | View and modify model configuration (**Context, Sampling, Token Limits**) |

> [!TIP]
> **Persistent Tuning:** Use `config` to set your model's personality (temp, top-p) and resource limits (n-ctx) once. All subsequent `run`, `chat`, and `batch` commands will inherit these settings automatically!

---

## ✨ Interface & Experience

The engine provides a professional-grade terminal experience:
- **GENERATION Block:** Immediate feedback on resolved parameters (Temperature, Max Tokens, Mode) before every query.
- **Dynamic Speed-Line:** High-accuracy `tokens/sec` and wall-clock time tracking for every generation.
- **Granular VRAM Backtracking:** Automatically finds the absolute maximum layers your GPU can handle (1-layer precision).
- **Segmented Model Support:** Native handling for multi-channel models like **GPT-OSS** (shows analysis/thinking phases clearly).

---

## 🏗️ Project structure

```text
model-accelerator/
├── engine.py          # The brain — loads models, runs inference, handles GPU detection
├── cli.py             # The interface — all the commands you actually type
├── pyproject.toml     # Python dependencies declaration
├── models.json        # Auto-generated — tracks which models you've downloaded (gitignored)
└── models/            # Where downloaded .gguf model files live (gitignored)
```

---

## 🏎️ Performance Tuning

Tuning your parameters based on your hardware is the key to maximizing tokens-per-second (tok/s). These flags can be added to any inference command (`run`, `chat`, `batch`, `bench`):

| Flag | Good Default | What it does |
|---|---|---|
| `--gpu-layers N` | `-1` (full offload) | Layers to offload to GPU. `0` = CPU-only, `-1` = all layers (as VRAM allows). |
| `--n-batch N` | `512` | Token batch size. Higher values increase throughput but use more VRAM. |
| `--n-threads N` | `6` to `8` | CPU threads for non-offloaded layers. *Diminishing returns beyond 8 threads.* |
| `--n-ctx N` | `4096` | Context window size. Larger sequences require exponentially more memory. |

**Example of an optimized run command:**
```bash
python cli.py run gptoss_20b --prompt "Explain quantum mechanics" --gpu-layers -1 --n-batch 1024
```

> `export LLM_GPU_LAYERS=-1`, `export LLM_N_BATCH=512`, `export LLM_N_THREADS=8`.

---

## 🎓 Deep Dive: How it Works

### 1. `engine.py` — The Brain
The core engine manages the lifecycle of the LLM. Key features include:
- **Zero-Reload Overlay:** The `Engine` instance keeps the model loaded in memory between requests to eliminate re-loading latency.
- **Smart GPU Detection:** Automatically picks the best offloading strategy based on available VRAM (supporting NVIDIA via CUDA, Apple Silicon via Metal/MPS, and CPU fallback).
- **Mmap Support:** Models use memory-mapping for instant startup times.

### 2. `cli.py` — The Interface
The CLI handles model acquisition and user interaction:
- **Seamless Loading:** Supports HuggingFace repos, direct URLs, and local files. It auto-selects the best quantization (prefers `Q4_K_M`) if not specified.
- **Inference Modes:** 
    - `run`: Streaming single-prompt generation.
    - `batch`: Bulk processing with per-prompt performance stats.
    - `chat`: Interactive REPL with sliding context window.
    - `bench`: Reproducible speed testing using `temp=0`.

---

## 🔮 Future Work

- [ ] **Continuous Batching:** Scheduler for running multiple sequences simultaneously.
- [ ] **OpenAI-Compatible Server:** Expose an HTTP endpoint using `llama_cpp.server`.
- [ ] **LoRA Support:** Native adapter loading in the CLI.

---

## 🗃️ Data Storage

Models are downloaded to `./models/` and aliased in a small JSON registry at `./models.json`:

```json
{
  "gptoss_20b": {
    "path": "./models/gptoss_20b.gguf",
    "source": "unsloth/gpt-oss-20b-GGUF"
  }
}
```

# model-accelerator

Minimal, high-performance LLM inference engine — **llama.cpp via GGUF models**.

---

## File layout

```
model-accelerator/
├── engine.py          # Core: Engine class + model registry helpers
├── cli.py             # CLI: subcommands (load / run / batch / chat / bench / list / info)
├── setup.sh           # Bootstrap: venv + llama-cpp-python install
├── requirements.txt   # Pip deps (llama-cpp-python, huggingface_hub)
├── models.json        # Auto-generated registry of downloaded models
└── models/            # Downloaded GGUF files (alias.gguf)
```

> **Hard constraint met:** exactly 2 Python files + 1 shell script.

---

## Setup (Arch Linux / any Linux)

```bash
# Clone / enter project dir
cd model-accelerator

# CPU-only (recommended to test first)
bash setup.sh

# NVIDIA GPU (cuBLAS — strongly recommended for throughput)
bash setup.sh --cuda

# Apple Silicon (Metal)
bash setup.sh --metal

# Activate venv
source .venv/bin/activate
```

> [!NOTE]
> `--cuda` compiles llama.cpp's cuBLAS kernels from source.  
> You'll need `cmake`, `gcc`/`g++`, and CUDA toolkit installed first.
> On Arch: `sudo pacman -S cmake cuda`

---

## Quick-start

```bash
source .venv/bin/activate

# 1. Download Mistral-7B Q4_K_M from HuggingFace
python cli.py load TheBloke/Mistral-7B-v0.1-GGUF --alias mistral7b

# 2. Run a single prompt
python cli.py run mistral7b --prompt "Explain quantum entanglement in 3 sentences."

# 3. Interactive chat
python cli.py chat mistral7b

# 4. Batch inference from a file
python cli.py batch mistral7b --file prompts.txt

# 5. Throughput benchmark
python cli.py bench mistral7b --runs 5
```

---

## All CLI commands

### `load` — Download & register a model

```bash
python cli.py load <source> --alias <name> [--file <gguf-filename>]
```

| Argument | Description |
|---|---|
| `source` | HF repo id (e.g. `TheBloke/Mistral-7B-v0.1-GGUF`), `hf://` URI, HTTPS URL, or local path |
| `--alias` | Short name to use for all other commands |
| `--file` | Specific `.gguf` filename inside the repo (auto-selected if omitted) |

**Auto-selection priority when `--file` is omitted:**  
`Q4_K_M` › `Q4_K_S` › `Q5_K_M` › `Q8_0` › first `.gguf` in repo

Examples:
```bash
# HuggingFace repo (auto-picks Q4_K_M)
python cli.py load TheBloke/Mistral-7B-v0.1-GGUF --alias mistral7b

# Explicit file
python cli.py load TheBloke/Mistral-7B-v0.1-GGUF \
  --alias mistral7b \
  --file mistral-7b-v0.1.Q4_K_M.gguf

# Direct URL
python cli.py load https://example.com/llama-3.gguf --alias llama3

# Local copy
python cli.py load /mnt/nas/phi-2.gguf --alias phi2
```

The model is saved to `./models/<alias>.gguf`.  
The registry `models.json` is updated automatically.

---

### `run` — Single-prompt inference

```bash
python cli.py run <alias> --prompt "text" [perf flags] [--stream]
```

```bash
python cli.py run mistral7b --prompt "Hello, who are you?"
python cli.py run mistral7b --prompt "Write me a haiku" --stream
python cli.py run mistral7b --prompt "Summarize X" --max-tokens 1024 --temp 0.3
```

---

### `batch` — Multi-prompt from file

```bash
python cli.py batch <alias> --file prompts.txt [--output results.json] [perf flags]
```

`prompts.txt` — one prompt per line, blank lines ignored.

```bash
python cli.py batch mistral7b --file prompts.txt --output results.json
```

Results JSON schema per entry:
```json
{
  "prompt": "...",
  "output": "...",
  "tokens": 312,
  "elapsed": 4.21,
  "tok_per_sec": 74.1
}
```

---

### `chat` — Interactive REPL

```bash
python cli.py chat <alias> [--system "You are..."] [perf flags]
```

```bash
python cli.py chat mistral7b
python cli.py chat mistral7b --system "You are a senior Python engineer."
```

Type `exit` or press `Ctrl-C` to quit.  
Each turn shows token count and tok/s.

---

### `bench` — Throughput benchmark

```bash
python cli.py bench <alias> [--prompt "..."] [--runs N] [perf flags]
```

```bash
python cli.py bench mistral7b --runs 5 --max-tokens 256
```

Runs the same prompt N times (temperature=0 for reproducibility) and prints:
- Per-run tok/s
- Average and peak tok/s
- Active tuning parameters

---

### `list` — Show all registered models

```bash
python cli.py list
```

### `info` — Show details for one alias

```bash
python cli.py info mistral7b
```

---

## Performance tuning flags

All `run / batch / chat / bench` commands accept these flags:

| Flag | Default | Effect |
|---|---|---|
| `--gpu-layers N` | auto | GPU layers to offload. `0` = CPU-only, `99` = all |
| `--n-ctx N` | 4096 | Context window size. Larger = more VRAM |
| `--n-batch N` | 512 | Token batch size. Larger = more throughput, more RAM |
| `--n-threads N` | min(cpus, 8) | CPU threads. Beyond 8 gives diminishing returns |
| `--max-tokens N` | 512 | Max tokens to generate |
| `--temp F` | 0.7 | Sampling temperature (0 = greedy/deterministic) |

### Throughput strategy explained

| Parameter | Why it matters |
|---|---|
| **`n_batch`** | Controls how many tokens are batched together per matrix-multiply. Larger batches amortize the CUDA kernel launch overhead. `512` is a good default; try `1024` on ≥12 GB VRAM. |
| **`n_gpu_layers`** | Each layer offloaded to GPU reduces CPU bottleneck drastically. On a 6 GB card, 20–28 layers of a 7B Q4_K_M model typically fit. Auto-detect selects 20. |
| **`n_threads`** | CPU threads for the un-offloaded layers. `llama.cpp`'s threading is efficient up to ~8; beyond that, lock contention can hurt. |
| **`use_mmap=True`** | Memory-maps the GGUF file → fast startup, avoids full RAM copy. Enabled unconditionally in engine. |
| **Model reuse** | The `Engine` object keeps the `Llama` instance alive between calls — no reload overhead per prompt. |

### 6 GB GPU recipe (7B Q4_K_M ≈ 4.1 GB file)

```bash
python cli.py run mistral7b \
  --prompt "..." \
  --gpu-layers 28 \
  --n-batch 512 \
  --n-threads 6
```

Expected throughput: **40–80 tok/s** depending on GPU.

---

## Environment variables

Set these for persistent tuning without flags:

```bash
export LLM_GPU_LAYERS=28
export LLM_N_CTX=4096
export LLM_N_BATCH=512
export LLM_N_THREADS=6

python cli.py run mistral7b --prompt "Hello"
```

---

## Example session

```
$ python cli.py load TheBloke/Mistral-7B-v0.1-GGUF --alias mistral7b
[load] Auto-selected: mistral-7b-v0.1.Q4_K_M.gguf  (from TheBloke/Mistral-7B-v0.1-GGUF)
[load] Downloading TheBloke/Mistral-7B-v0.1-GGUF/mistral-7b-v0.1.Q4_K_M.gguf …
[load] Saved → models/mistral7b.gguf  (4.37 GB)
[registry] 'mistral7b' → models/mistral7b.gguf

$ python cli.py bench mistral7b --runs 3 --max-tokens 128
[engine] Loading 'mistral7b' …
         path       : models/mistral7b.gguf
         n_ctx      : 4096
         n_batch    : 512
         n_threads  : 8
         n_gpu_layers: 20
[engine] Ready.

[bench] 3 runs × prompt='Tell me a short story about a robot.'  max_tokens=128

  Run  1: 128 tokens  61.3 tok/s  (2.09s)
  Run  2: 128 tokens  63.1 tok/s  (2.03s)
  Run  3: 128 tokens  62.8 tok/s  (2.04s)

[bench] Average: 62.4 tok/s   Peak: 63.1 tok/s
[bench] n_batch=512  n_threads=8  n_gpu_layers=20

$ python cli.py chat mistral7b

[chat] Model: mistral7b  (type 'exit' or Ctrl-C to quit)

You: Who are you?
Assistant:  I am Mistral, a large language model...
  (47 tokens, 59.2 tok/s)

You: exit
[chat] Goodbye.
```

---

## models.json format

```json
{
  "mistral7b": {
    "path": "./models/mistral7b.gguf",
    "source": "TheBloke/Mistral-7B-v0.1-GGUF"
  },
  "phi2": {
    "path": "./models/phi2.gguf",
    "source": "/mnt/nas/phi-2.gguf"
  }
}
```

---

## Future work

```python
# TODO: add scheduler + pseudo-continuous batching
```

- True continuous batching (run multiple sequences in parallel inside one `Llama` call)  
- OpenAI-compatible HTTP server (`llama_cpp.server`)  
- Quantization comparison table per alias  
- LoRA adapter support  

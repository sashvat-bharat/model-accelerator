"""
engine.py — Core inference engine for llama.cpp (GGUF) models.

Handles:
  - Model registry (models.json)
  - Model loading with GPU/CPU auto-detection
  - Batched and single-prompt inference
  - Performance tuning (n_batch, n_threads, n_gpu_layers)

# TODO: add scheduler + pseudo-continuous batching
"""

from __future__ import annotations

import json
import os
import sys
import time
import multiprocessing
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

MODELS_DIR = Path("./models")
REGISTRY_FILE = Path("./models.json")


def _load_registry() -> dict:
    if REGISTRY_FILE.exists():
        with REGISTRY_FILE.open() as f:
            return json.load(f)
    return {}


def _save_registry(reg: dict) -> None:
    REGISTRY_FILE.write_text(json.dumps(reg, indent=2))


def register_model(alias: str, path: str, source: str) -> None:
    reg = _load_registry()
    reg[alias] = {"path": path, "source": source}
    _save_registry(reg)
    print(f"[registry] '{alias}' → {path}")


def resolve_model(alias: str) -> str:
    reg = _load_registry()
    if alias not in reg:
        sys.exit(f"[error] Unknown alias '{alias}'. Run:  engine load <url> --alias {alias}")
    path = reg[alias]["path"]
    if not Path(path).exists():
        sys.exit(f"[error] Model file missing: {path}")
    return path


def list_models() -> dict:
    return _load_registry()


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _detect_gpu_layers() -> int:
    """
    Returns a reasonable n_gpu_layers value:
      - 0  → CPU-only fallback
      - 35 → conservative default for a 6 GB VRAM card   (covers most 7B Q4 models)
      - 99 → effectively 'all layers' for large VRAM cards

    Detection strategy (in order):
      1. Env override  LLM_GPU_LAYERS=<int>
      2. PyTorch CUDA  (if available)
      3. subprocess nvidia-smi / rocm-smi
      4. Fall back to 0 (CPU)
    """
    env = os.environ.get("LLM_GPU_LAYERS")
    if env is not None:
        return int(env)

    # PyTorch path
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram_gb >= 20:
                return 99
            elif vram_gb >= 8:
                return 35
            else:
                return 20  # 6 GB card — partial offload
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return 1  # Metal — llama.cpp Metal backend handles it
    except ImportError:
        pass

    # nvidia-smi fallback
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        )
        vram_mb = int(out.decode().strip().split("\n")[0])
        vram_gb = vram_mb / 1024
        if vram_gb >= 20:
            return 99
        elif vram_gb >= 8:
            return 35
        else:
            return 20
    except Exception:
        pass

    print("[info] No GPU detected — running on CPU (set LLM_GPU_LAYERS=<n> to override)")
    return 0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

# Performance defaults — tunable via env vars or constructor kwargs
_DEFAULT_N_CTX    = int(os.environ.get("LLM_N_CTX",    "4096"))
_DEFAULT_N_BATCH  = int(os.environ.get("LLM_N_BATCH",  "512"))   # larger = more throughput
_DEFAULT_N_THREADS = int(os.environ.get("LLM_N_THREADS", str(min(multiprocessing.cpu_count(), 8))))


class Engine:
    """
    Thin wrapper around llama-cpp-python's Llama class.

    Design decisions for throughput:
      - Single model instance; never reloaded between calls.
      - n_batch=512 (default) — token-batch size fed to the model at once.
        Increasing this trades memory for throughput on long prompts.
      - n_threads=min(cpus, 8) — beyond ~8 threads, llama.cpp sees diminishing returns.
      - n_gpu_layers auto-detected — partial offloading beats full CPU for most 6 GB cards.
      - verbose=False — suppresses llama.cpp progress spam.
    """

    def __init__(
        self,
        alias: str,
        n_ctx: int = _DEFAULT_N_CTX,
        n_batch: int = _DEFAULT_N_BATCH,
        n_threads: int = _DEFAULT_N_THREADS,
        n_gpu_layers: int | None = None,
    ) -> None:
        from llama_cpp import Llama  # lazy import — lets the CLI load fast

        model_path = str(Path(resolve_model(alias)).resolve())
        gpu_layers = n_gpu_layers if n_gpu_layers is not None else _detect_gpu_layers()

        print(f"[engine] Loading '{alias}' …")
        print(f"         path       : {model_path}")
        print(f"         n_ctx      : {n_ctx}")
        print(f"         n_batch    : {n_batch}")
        print(f"         n_threads  : {n_threads}")
        print(f"         n_gpu_layers: {gpu_layers}")

        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            n_gpu_layers=gpu_layers,
            verbose=False,
            logits_all=False,
            use_mmap=True,
        )
        self._alias = alias
        print(f"[engine] Ready.\n")

    # ------------------------------------------------------------------
    # Single-prompt generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """
        Generate text for a single prompt.

        Args:
            prompt:     Input text.
            max_tokens: Maximum tokens to generate.
            temperature / top_p: Sampling parameters.
            stream:     If True, yields tokens as a generator.

        Returns:
            Complete string (stream=False) or token iterator (stream=True).
        """
        kwargs = dict(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=False,
            stream=stream,
        )

        if stream:
            def _gen():
                for chunk in self._llm(**kwargs):
                    yield chunk["choices"][0]["text"]
            return _gen()

        result = self._llm(**kwargs)
        return result["choices"][0]["text"]

    # ------------------------------------------------------------------
    # Batch generation (loop-based; good enough for local inference)
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> list[dict]:
        """
        Run inference over a list of prompts sequentially, reusing the model
        instance.  Reports per-prompt and aggregate tokens/sec.

        Returns a list of dicts:
          {"prompt": str, "output": str, "tokens": int, "elapsed": float, "tok_per_sec": float}
        """
        results = []
        total_tokens = 0
        wall_start = time.perf_counter()

        for i, prompt in enumerate(prompts, 1):
            t0 = time.perf_counter()
            raw = self._llm(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                echo=False,
                stream=False,
            )
            elapsed = time.perf_counter() - t0

            text = raw["choices"][0]["text"]
            usage = raw.get("usage", {})
            # llama-cpp-python populates usage when echo=False
            gen_tokens = usage.get("completion_tokens", len(text.split()))

            tok_per_sec = gen_tokens / elapsed if elapsed > 0 else 0.0
            total_tokens += gen_tokens

            results.append({
                "prompt": prompt,
                "output": text,
                "tokens": gen_tokens,
                "elapsed": round(elapsed, 3),
                "tok_per_sec": round(tok_per_sec, 1),
            })
            print(f"  [{i}/{len(prompts)}] {gen_tokens} tokens  {tok_per_sec:.1f} tok/s")

        wall = time.perf_counter() - wall_start
        agg = total_tokens / wall if wall > 0 else 0.0
        print(f"\n[batch] {total_tokens} total tokens in {wall:.2f}s → {agg:.1f} tok/s aggregate")
        return results

    # ------------------------------------------------------------------
    # Interactive chat (simple rolling context window)
    # ------------------------------------------------------------------

    def chat(
        self,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> None:
        """
        Simple REPL chat loop.
        """
        history = f"[SYSTEM]: {system_prompt}\n"
        print(f"\n\033[94m[chat]\033[0m Model: \033[1m{self._alias}\033[0m  (type 'exit' to quit)\n")

        while True:
            try:
                user_input = input("\033[32mYou:\033[0m ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n[chat] Goodbye.")
                break

            if user_input.lower() in {"exit", "quit", "q"}:
                print("[chat] Goodbye.")
                break
            if not user_input:
                continue

            history += f"\n[USER]: {user_input}\n[ASSISTANT]:"

            tokens_out = []
            print("\033[34mAssistant:\033[0m ", end="", flush=True)
            t0 = time.perf_counter()

            for chunk in self._llm(
                prompt=history,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                echo=False,
                stream=True,
                stop=["[USER]:", "[SYSTEM]:"],
            ):
                tok = chunk["choices"][0]["text"]
                print(tok, end="", flush=True)
                tokens_out.append(tok)

            elapsed = time.perf_counter() - t0
            response = "".join(tokens_out)
            history += response
            n_tok = len(tokens_out)
            tps = n_tok / elapsed if elapsed > 0 else 0
            print(f"\n\033[90m[{n_tok} tokens | {tps:.1f} tok/s]\033[0m\n")

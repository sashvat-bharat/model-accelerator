"""
engine.py — Core inference engine for GGUF models.

Handles:
  - Model registry (models.json)
  - Native GGUF metadata parsing (Chat templates, EOG tokens, n_ctx, block_count)
  - True multi-sequence parallel inference via C API bindings
  - Auto GPU offload detection and self-healing memory allocation
"""

from __future__ import annotations

import json
import os
import sys
import time
import gc
import threading
import multiprocessing
from pathlib import Path
from typing import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, asdict

from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# Custom exception to replace sys.exit() in library functions
# ---------------------------------------------------------------------------


class ModelNotFoundError(Exception):
    pass


# ---------------------------------------------------------------------------
# Suppress C stderr (thread-safe, leak-free)
# ---------------------------------------------------------------------------

_stderr_lock = threading.Lock()


@contextmanager
def suppress_c_stderr():
    with _stderr_lock:
        fd = sys.stderr.fileno()
        original_stderr = os.dup(fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, fd)
        try:
            yield
        finally:
            os.dup2(original_stderr, fd)
            os.close(original_stderr)
            os.close(devnull)


# ---------------------------------------------------------------------------
# One-time global __del__ patch (fix #16)
# ---------------------------------------------------------------------------


def _apply_safe_del_patch():
    import llama_cpp._internals

    if hasattr(llama_cpp._internals.LlamaModel.__del__, "_safe_patched"):
        return
    original_del = llama_cpp._internals.LlamaModel.__del__

    def safe_del(obj):
        try:
            original_del(obj)
        except Exception:
            pass

    safe_del._safe_patched = True
    llama_cpp._internals.LlamaModel.__del__ = safe_del


_apply_safe_del_patch()


# ---------------------------------------------------------------------------
# Default Inference & Sampling Parameters
# ---------------------------------------------------------------------------

_DEFAULT_N_CTX = int(os.environ.get("LLM_N_CTX", "4096"))
_DEFAULT_N_BATCH = int(os.environ.get("LLM_N_BATCH", "512"))
_DEFAULT_N_THREADS = int(
    os.environ.get("LLM_N_THREADS", str(min(multiprocessing.cpu_count(), 8)))
)
_MAX_THREADS_BATCH = multiprocessing.cpu_count()

_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_TOP_P = 1.0
_DEFAULT_TOP_K = 0
_DEFAULT_MAX_TOKENS = "auto"

# ---------------------------------------------------------------------------
# Dataclasses & Configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelCapabilities:
    alias: str
    source: str
    path: str
    exists: bool
    size: str
    total_context_length: int
    total_layers: int


@dataclass
class ModelConfig:
    configured_context_length: int | None = None
    max_input_tokens: str | int = _DEFAULT_MAX_TOKENS
    max_output_tokens: str | int = _DEFAULT_MAX_TOKENS
    temperature: float = _DEFAULT_TEMPERATURE
    top_p: float = _DEFAULT_TOP_P
    top_k: int = _DEFAULT_TOP_K


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

_ENV_DIR = os.environ.get("GGUF_MODELS_DIR")
if _ENV_DIR:
    MODELS_DIR = Path(_ENV_DIR)
    REGISTRY_FILE = MODELS_DIR / "models.json"
else:
    MODELS_DIR = Path("./models")
    REGISTRY_FILE = Path("./models.json")


def _load_registry() -> dict:
    if REGISTRY_FILE.exists():
        with REGISTRY_FILE.open() as f:
            return json.load(f)
    return {}


def _save_registry(reg: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(reg, indent=2))


def register_model(alias: str, path: str, source: str) -> None:
    reg = _load_registry()
    reg[alias] = {"path": path, "source": source}
    _save_registry(reg)
    console.print(f"[dim green][registry][/] '{alias}' → {path}")


def resolve_model(alias: str) -> str:
    reg = _load_registry()
    if alias not in reg:
        raise ModelNotFoundError(
            f"Unknown alias '{alias}'. Run: python cli.py load <url> --alias {alias}"
        )

    path = Path(reg[alias]["path"])
    if not path.exists():
        raise ModelNotFoundError(f"Model path missing: {path}")

    if path.is_dir():
        ggufs = list(path.glob("*.gguf"))
        if not ggufs:
            raise ModelNotFoundError(f"No .gguf file found in directory: {path}")
        return str(ggufs[0])

    return str(path)


def list_models() -> dict:
    return _load_registry()


def get_model_capabilities(alias: str) -> ModelCapabilities:
    """Read native GGUF metadata directly to determine limits & layers."""
    from llama_cpp import Llama

    reg = _load_registry().get(alias, {})
    path_str = reg.get("path", "")
    p = Path(path_str) if path_str else Path()

    caps = ModelCapabilities(
        alias=alias,
        source=reg.get("source", "local"),
        path=str(p),
        exists=p.exists(),
        size=f"{p.stat().st_size / 1e9:.2f} GB" if p.exists() else "0 GB",
        total_context_length=_DEFAULT_N_CTX,
        total_layers=32,
    )

    if p.exists():
        try:
            actual_file = resolve_model(alias)
            with suppress_c_stderr():
                temp_llm = Llama(model_path=actual_file, vocab_only=True, verbose=False)

            for k, v in temp_llm.metadata.items():
                if k.endswith(".context_length") and not k.endswith(
                    ".original_context_length"
                ):
                    try:
                        caps.total_context_length = int(v)
                    except (ValueError, TypeError):
                        pass

                if k.endswith(".block_count"):
                    try:
                        caps.total_layers = int(v)
                    except (ValueError, TypeError):
                        pass

            del temp_llm
        except ModelNotFoundError:
            raise
        except Exception as e:
            console.print(
                f"[dim yellow][warn] Could not read GGUF metadata for '{alias}': {e}[/]"
            )

    return caps


def get_model_config(alias: str) -> ModelConfig:
    reg = _load_registry()
    if alias not in reg:
        return ModelConfig()

    model_dir = Path(reg[alias]["path"])
    params_path = model_dir / "params.json"

    if params_path.exists():
        try:
            with params_path.open() as f:
                data = json.load(f)
                valid_keys = {k for k in data if hasattr(ModelConfig, k)}
                filtered_data = {k: data[k] for k in valid_keys}
                return ModelConfig(**filtered_data)
        except Exception:
            pass

    return ModelConfig()


def set_model_config(alias: str, config: ModelConfig) -> None:
    reg = _load_registry()
    if alias not in reg:
        return

    model_dir = Path(reg[alias]["path"])
    model_dir.mkdir(parents=True, exist_ok=True)
    params_path = model_dir / "params.json"

    params_path.write_text(json.dumps(asdict(config), indent=2))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class Engine:
    def __init__(
        self,
        alias: str,
        n_ctx: int | None = None,
        n_batch: int | None = None,
        n_threads: int | None = None,
        n_gpu_layers: int | str | None = None,
    ) -> None:
        from llama_cpp import Llama

        model_path = resolve_model(alias)
        config = get_model_config(alias)
        caps = get_model_capabilities(alias)

        if n_ctx is None:
            n_ctx = config.configured_context_length or _DEFAULT_N_CTX
        n_ctx = min(n_ctx, caps.total_context_length)

        if n_batch is None:
            n_batch = _DEFAULT_N_BATCH
        if n_threads is None:
            n_threads = _DEFAULT_N_THREADS

        exact_model_layers = caps.total_layers + 1

        if n_gpu_layers == "max" or n_gpu_layers is None:
            gpu_layers = exact_model_layers
            is_auto = True
        else:
            gpu_layers = int(n_gpu_layers)
            is_auto = False

        self._config = config
        self._caps = caps
        self._alias = alias
        self._n_ctx = n_ctx

        console.print(f"[bold cyan][engine][/] Loading '{alias}' …")
        console.print(f"         path       : [dim]{model_path}[/]")
        console.print(f"         n_ctx      : [yellow]{n_ctx}[/]")
        console.print(f"         n_batch    : [yellow]{n_batch}[/]")
        console.print(
            f"         n_threads  : {n_threads} (Eval) / {_MAX_THREADS_BATCH} (Prefill)"
        )
        console.print(
            f"         gpu_layers : [green]{gpu_layers}[/] {'(Auto-Max)' if is_auto else '(Forced)'}"
        )

        has_gpu = gpu_layers > 0

        self._llm = None
        max_attempts = min(gpu_layers + 1, 20)
        attempts = 0
        hit_oom = False

        while self._llm is None and attempts < max_attempts:
            try:
                with suppress_c_stderr():
                    self._llm = Llama(
                        model_path=model_path,
                        n_ctx=n_ctx,
                        n_batch=n_batch,
                        n_threads=n_threads,
                        n_threads_batch=_MAX_THREADS_BATCH,
                        n_gpu_layers=gpu_layers,
                        verbose=False,
                        use_mmap=True,
                        flash_attn=has_gpu,
                    )

                if hit_oom and gpu_layers > 0:
                    margin = max(2, int(exact_model_layers * 0.06))
                    safe_layers = max(0, gpu_layers - margin)

                    console.print(
                        f"[dim yellow]  → 100% VRAM saturation detected at {gpu_layers} layers.[/]"
                    )
                    console.print(
                        f"[dim yellow]  → Applying {margin}-layer compute margin. Reloading at {safe_layers}...[/]"
                    )

                    del self._llm
                    self._llm = None
                    gc.collect()
                    time.sleep(0.3)

                    gpu_layers = safe_layers
                    hit_oom = False
                    continue

            except (ValueError, RuntimeError) as e:
                attempts += 1
                hit_oom = True
                if gpu_layers <= 0:
                    raise RuntimeError(
                        f"Engine failed to load even on CPU. Out of RAM? Error: {e}"
                    )

                old_layers = gpu_layers
                gpu_layers -= 1

                if is_auto:
                    console.print(
                        f"[yellow][warn][/] VRAM limit reached at {old_layers} layers. Retrying with {gpu_layers}..."
                    )
                else:
                    console.print(
                        f"[yellow][warn][/] Forced {old_layers} layers failed (OOM). Self-healing to {gpu_layers}..."
                    )

                time.sleep(0.1)

        if not self._llm:
            raise RuntimeError(
                f"Failed to load engine for '{alias}'. Exhausted all {max_attempts} attempts."
            )

        if self._llm.metadata.get("tokenizer.chat_template"):
            console.print(
                "[dim green][engine] Native Chat Template detected in GGUF metadata.[/]"
            )
        else:
            console.print(
                "[yellow][warn] No native chat template found in GGUF. Using fallback.[/]"
            )

        console.print("[bold green][engine] Ready.[/]\n")

    def apply_chat_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        """Evaluates native GGUF internal Jinja template."""
        template = self._llm.metadata.get("tokenizer.chat_template")
        if template:
            try:
                import jinja2

                env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)

                def _raise_err(msg):
                    raise ValueError(msg)

                env.globals["raise_exception"] = _raise_err

                bos_token = (
                    self._llm.detokenize([self._llm.token_bos()]).decode("utf-8")
                    if self._llm.token_bos() != -1
                    else ""
                )
                eos_token = (
                    self._llm.detokenize([self._llm.token_eos()]).decode("utf-8")
                    if self._llm.token_eos() != -1
                    else ""
                )

                render_kwargs = dict(
                    messages=messages,
                    bos_token=bos_token,
                    eos_token=eos_token,
                    add_generation_prompt=add_generation_prompt,
                )
                if enable_thinking:
                    render_kwargs["enable_thinking"] = True

                return env.from_string(template).render(**render_kwargs)
            except Exception as e:
                console.print(
                    f"[yellow][warn][/] Failed to evaluate native template: {e}. Falling back."
                )

        prompt = ""
        for m in messages:
            prompt += f"[{m['role'].upper()}]: {m['content']}\n"
        if add_generation_prompt:
            prompt += "[ASSISTANT]:"
        return prompt

    def _detect_think_capable(self) -> bool:
        """Check if the model's chat template supports thinking mode."""
        template = self._llm.metadata.get("tokenizer.chat_template", "")
        if not template:
            return False
        think_markers = ["<think>", "thinking_prefix", "thinking_suffix"]
        return any(marker in template for marker in think_markers)

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stream: bool = False,
        chat_format: bool = False,
        think: bool = False,
    ) -> str | Iterator[str]:

        temperature = (
            temperature if temperature is not None else self._config.temperature
        )
        top_p = top_p if top_p is not None else self._config.top_p
        top_k = top_k if top_k is not None else self._config.top_k

        if max_tokens is None:
            saved = self._config.max_output_tokens
            if saved == "auto":
                input_toks = self._llm.tokenize(prompt.encode("utf-8"))
                max_tokens = max(128, self._n_ctx - len(input_toks) - 10)
                max_tokens = min(max_tokens, self._n_ctx - len(input_toks) - 10)
            else:
                max_tokens = int(saved)

        use_thinking = think and chat_format and self._detect_think_capable()

        if chat_format and not use_thinking:
            messages = [{"role": "user", "content": prompt}]
            if stream:

                def _gen():
                    for chunk in self._llm.create_chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        stream=True,
                    ):
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content

                return _gen()

            result = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            return result["choices"][0]["message"].get("content", "")

        if use_thinking:
            prompt_text = self.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                enable_thinking=True,
            )
        else:
            prompt_text = prompt

        kwargs = dict(
            prompt=prompt_text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            echo=False,
            stream=stream,
        )

        if stream:

            def _gen():
                for chunk in self._llm(**kwargs):
                    yield chunk["choices"][0]["text"]

            return _gen()

        return self._llm(**kwargs)["choices"][0]["text"]

    # ------------------------------------------------------------------
    # Parallel Batch Generation (Refactored)
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        parallel: int = 0,
        chat_format: bool = False,
        think: bool = False,
    ) -> list[dict]:

        temperature = (
            temperature if temperature is not None else self._config.temperature
        )
        top_p = top_p if top_p is not None else self._config.top_p
        top_k = top_k if top_k is not None else self._config.top_k

        if max_tokens is None:
            saved = self._config.max_output_tokens
            max_tokens = min(2048, self._n_ctx // 4) if saved == "auto" else int(saved)

        all_results: list[dict] = []
        total_tokens = 0
        wall_start = time.perf_counter()
        done_count = 0

        tokenized = []
        use_thinking = think and chat_format and self._detect_think_capable()
        for p in prompts:
            if chat_format:
                actual_prompt = self.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True,
                    enable_thinking=use_thinking,
                )
            else:
                actual_prompt = p
            tokenized.append(
                self._llm.tokenize(actual_prompt.encode("utf-8"), add_bos=True)
            )

        sub_batches = self._calculate_sub_batches(tokenized, max_tokens, parallel)
        n_sub = len(sub_batches)

        if n_sub > 1:
            console.print(
                f"[dim cyan][batch][/] Split into {n_sub} sub-batches (parallel={'auto' if parallel == 0 else parallel})"
            )

        for sb_i, sb_indices in enumerate(sub_batches):
            sb_prompts = [prompts[i] for i in sb_indices]
            sb_tokens = [tokenized[i] for i in sb_indices]

            if n_sub > 1:
                console.print(
                    f"\n[bold cyan][batch][/] Sub-batch {sb_i + 1}/{n_sub}: {len(sb_indices)} prompts parallel"
                )

            sb_results = self._parallel_decode(
                sb_prompts,
                sb_tokens,
                max_tokens,
                temperature,
                top_p,
                top_k,
                start_idx=done_count,
                total=len(prompts),
            )

            for r in sb_results:
                total_tokens += r["tokens"]
            all_results.extend(sb_results)
            done_count += len(sb_indices)

        wall = time.perf_counter() - wall_start
        agg = total_tokens / wall if wall > 0 else 0.0
        console.print(
            f"\n[bold green][batch][/] {total_tokens} total tokens in {wall:.2f}s → {agg:.1f} tok/s aggregate"
        )
        return all_results

    def _calculate_sub_batches(self, tokenized, max_tokens, parallel):
        if parallel > 0:
            return [
                list(range(start, min(start + parallel, len(tokenized))))
                for start in range(0, len(tokenized), parallel)
            ]

        sub_batches, current_batch, current_ctx_usage = [], [], 0
        n_ctx = self._llm._n_ctx

        for idx, toks in enumerate(tokenized):
            seq_budget = len(toks) + max_tokens
            if seq_budget > n_ctx:
                if current_batch:
                    sub_batches.append(current_batch)
                if len(toks) > n_ctx:
                    raise RuntimeError(
                        f"Prompt {idx} requires {len(toks)} tokens but context window is {n_ctx}. "
                        f"Reduce --n-ctx or split the prompt."
                    )
                sub_batches.append([idx])
                current_batch, current_ctx_usage = [], 0
                continue
            if current_ctx_usage + seq_budget > n_ctx and current_batch:
                sub_batches.append(current_batch)
                current_batch, current_ctx_usage = [], 0
            current_batch.append(idx)
            current_ctx_usage += seq_budget

        if current_batch:
            sub_batches.append(current_batch)
        return sub_batches

    def _parallel_decode(
        self,
        prompts,
        prompt_tokens,
        max_tokens,
        temperature,
        top_p,
        top_k,
        start_idx,
        total,
    ) -> list[dict]:
        import llama_cpp.llama_cpp as ll

        llm = self._llm
        n_seq = len(prompts)
        vocab = llm._model.vocab

        batch_ctx, samplers = self._init_parallel_context(
            ll, prompt_tokens, max_tokens, temperature, top_p, top_k
        )

        generated: list[list[int]] = [[] for _ in range(n_seq)]
        seq_pos: list[int] = [0] * n_seq
        finished: list[bool] = [False] * n_seq

        t0 = time.perf_counter()

        try:
            last_tokens = self._parallel_phase1_prefill(
                ll, batch_ctx, prompt_tokens, seq_pos
            )
            self._parallel_phase1_5_first_decode(
                ll, batch_ctx, samplers, vocab, last_tokens, generated, finished
            )
            self._parallel_phase2_autoregressive(
                ll, batch_ctx, samplers, vocab, max_tokens, seq_pos, generated, finished
            )
        finally:
            for s in samplers:
                s.close()
            ll.llama_free(batch_ctx)

        elapsed = time.perf_counter() - t0

        results = []
        for seq_id in range(n_seq):
            text = llm.detokenize(generated[seq_id]).decode("utf-8", errors="replace")
            n_tok = len(generated[seq_id])
            seq_tok_per_sec = n_tok / elapsed if elapsed > 0 else 0.0
            results.append(
                {
                    "prompt": prompts[seq_id],
                    "output": text,
                    "tokens": n_tok,
                    "elapsed": round(elapsed, 3),
                    "tok_per_sec": round(seq_tok_per_sec, 1),
                }
            )
            console.print(
                f"  [dim][{start_idx + seq_id + 1}/{total}][/] {n_tok} tokens  (parallel batch)"
            )

        total_gen = sum(len(g) for g in generated)
        tok_per_sec = total_gen / elapsed if elapsed > 0 else 0.0
        console.print(
            f"  → decoded {n_seq} prompts in {elapsed:.2f}s — {total_gen} tokens total — [bold]{tok_per_sec:.1f} tok/s[/] effective"
        )

        return results

    def _init_parallel_context(
        self, ll, prompt_tokens, max_tokens, temperature, top_p, top_k
    ):
        llm = self._llm
        n_seq = len(prompt_tokens)
        required_ctx = sum(len(t) + max_tokens for t in prompt_tokens)

        if required_ctx > 128 * 1024:
            raise RuntimeError(
                f"Batch requires {required_ctx} context tokens, exceeding safe limit of 131072. "
                f"Reduce --parallel or --max-tokens."
            )

        ctx_params = ll.llama_context_default_params()
        ctx_params.n_ctx = max(required_ctx, llm.context_params.n_ctx)
        ctx_params.n_batch = max(llm.context_params.n_batch, n_seq)
        ctx_params.n_seq_max = n_seq
        ctx_params.n_threads = llm.context_params.n_threads
        ctx_params.n_threads_batch = _MAX_THREADS_BATCH

        for attr in ["flash_attn", "flash_attn_type", "attention_type", "offload_kqv"]:
            if hasattr(llm.context_params, attr):
                setattr(ctx_params, attr, getattr(llm.context_params, attr))

        batch_ctx = ll.llama_init_from_model(llm.model, ctx_params)
        if batch_ctx is None:
            raise RuntimeError("Failed to create batch context. VRAM exhausted.")

        samplers = [
            llm._init_sampler(
                temp=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=0.05,
                repeat_penalty=1.1,
                frequency_penalty=0.1,
                presence_penalty=0.1,
            )
            for _ in range(n_seq)
        ]

        return batch_ctx, samplers

    def _parallel_phase1_prefill(self, ll, batch_ctx, prompt_tokens, seq_pos):
        non_last_prefill, last_tokens = [], []
        n_batch = self._llm.n_batch

        for seq_id, toks in enumerate(prompt_tokens):
            if len(toks) == 0:
                last_tokens.append((self._llm.token_bos(), 0, seq_id))
                seq_pos[seq_id] = 1
                continue
            for i, tok in enumerate(toks[:-1]):
                non_last_prefill.append((tok, i, seq_id))
            last_tokens.append((toks[-1], len(toks) - 1, seq_id))
            seq_pos[seq_id] = len(toks)

        for chunk_start in range(0, len(non_last_prefill), n_batch):
            chunk = non_last_prefill[chunk_start : chunk_start + n_batch]
            batch = ll.llama_batch_init(len(chunk), 0, 1)
            batch.n_tokens = len(chunk)
            for j, (tok, pos, seq_id) in enumerate(chunk):
                batch.token[j], batch.pos[j], batch.n_seq_id[j] = tok, pos, 1
                batch.seq_id[j][0], batch.logits[j] = seq_id, False
            ll.llama_decode(batch_ctx, batch)
            ll.llama_batch_free(batch)

        return last_tokens

    def _parallel_phase1_5_first_decode(
        self, ll, batch_ctx, samplers, vocab, last_tokens, generated, finished
    ):
        n_seq = len(last_tokens)
        batch = ll.llama_batch_init(n_seq, 0, 1)
        batch.n_tokens = n_seq

        for j, (tok, pos, seq_id) in enumerate(last_tokens):
            batch.token[j], batch.pos[j], batch.n_seq_id[j] = tok, pos, 1
            batch.seq_id[j][0], batch.logits[j] = seq_id, True

        ll.llama_decode(batch_ctx, batch)

        for seq_i in range(n_seq):
            seq_id = last_tokens[seq_i][2]
            token = ll.llama_sampler_sample(samplers[seq_id].sampler, batch_ctx, seq_i)
            ll.llama_sampler_accept(samplers[seq_id].sampler, token)

            if ll.llama_vocab_is_eog(vocab, token):
                finished[seq_id] = True
            else:
                generated[seq_id].append(token)

        ll.llama_batch_free(batch)

    def _parallel_phase2_autoregressive(
        self, ll, batch_ctx, samplers, vocab, max_tokens, seq_pos, generated, finished
    ):
        n_seq = len(generated)
        for step in range(1, max_tokens):
            active = [
                s
                for s in range(n_seq)
                if not finished[s] and len(generated[s]) < max_tokens
            ]
            if not active:
                break

            batch = ll.llama_batch_init(len(active), 0, 1)
            batch.n_tokens = len(active)
            for j, seq_id in enumerate(active):
                batch.token[j], batch.pos[j], batch.n_seq_id[j] = (
                    generated[seq_id][-1],
                    seq_pos[seq_id],
                    1,
                )
                batch.seq_id[j][0], batch.logits[j] = seq_id, True
                seq_pos[seq_id] += 1

            if ll.llama_decode(batch_ctx, batch) != 0:
                ll.llama_batch_free(batch)
                break

            for j, seq_id in enumerate(active):
                token = ll.llama_sampler_sample(samplers[seq_id].sampler, batch_ctx, j)
                ll.llama_sampler_accept(samplers[seq_id].sampler, token)

                if ll.llama_vocab_is_eog(vocab, token):
                    finished[seq_id] = True
                else:
                    generated[seq_id].append(token)

            ll.llama_batch_free(batch)

    # ------------------------------------------------------------------
    # Interactive chat (REPL)
    # ------------------------------------------------------------------

    def _count_message_tokens(self, messages: list[dict]) -> int:
        total = 0
        for m in messages:
            content = m.get("content", "")
            total += len(self._llm.tokenize(content.encode("utf-8")))
        return total

    def _truncate_chat_history(self, messages: list[dict]) -> list[dict]:
        if len(messages) <= 1:
            return messages

        system_msg = messages[0] if messages[0]["role"] == "system" else None
        conversation = messages[1:] if system_msg else messages[:]

        while len(conversation) > 2:
            tokens = self._count_message_tokens(conversation)
            if tokens < self._n_ctx * 0.8:
                break
            conversation = conversation[2:]

        if system_msg:
            return [system_msg] + conversation
        return conversation

    def chat(
        self,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        think: bool = False,
    ) -> None:

        temperature = (
            temperature if temperature is not None else self._config.temperature
        )
        top_p = top_p if top_p is not None else self._config.top_p
        top_k = top_k if top_k is not None else self._config.top_k

        if max_tokens is None:
            saved = self._config.max_output_tokens
            max_tokens = min(4096, self._n_ctx // 2) if saved == "auto" else int(saved)

        think_enabled = think and self._detect_think_capable()
        if think_enabled:
            console.print(
                f"\n[bold blue][chat][/] Model: [bold]{self._alias}[/]  (type 'exit' to quit) [dim cyan](thinking enabled)[/]\n"
            )
        else:
            console.print(
                f"\n[bold blue][chat][/] Model: [bold]{self._alias}[/]  (type 'exit' to quit)\n"
            )

        messages = [{"role": "system", "content": system_prompt}]

        while True:
            try:
                user_input = console.input("[bold green]You:[/] ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Goodbye.[/]")
                break

            if user_input.lower() in {"exit", "quit", "q"}:
                console.print("[dim]Goodbye.[/]")
                break
            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})
            messages = self._truncate_chat_history(messages)
            tokens_out = []
            console.print("[bold blue]Assistant:[/] ", end="")
            t0 = time.perf_counter()

            if think_enabled:
                prompt_text = self.apply_chat_template(
                    messages, add_generation_prompt=True, enable_thinking=True
                )
                thinking_started = False
                try:
                    for chunk in self._llm(
                        prompt=prompt_text,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        stream=True,
                    ):
                        tok = chunk["choices"][0]["text"]
                        if "<think>" in tok:
                            thinking_started = True
                            parts = tok.split("<think>", 1)
                            if parts[0]:
                                print(parts[0], end="", flush=True)
                            console.print(f"[bold magenta]<think>[/]")
                            if parts[1]:
                                console.print(
                                    f"[dim cyan]{parts[1]}[/]", end="", flush=True
                                )
                                tokens_out.append(parts[1])
                            continue
                        elif "</think>" in tok:
                            parts = tok.split("</think>", 1)
                            if parts[0]:
                                console.print(f"[dim cyan]{parts[0]}[/]")
                                tokens_out.append(parts[0])
                            console.print(f"[bold magenta]</think>[/]")
                            if parts[1]:
                                print(parts[1], end="", flush=True)
                                tokens_out.append(parts[1])
                            continue
                        elif not thinking_started:
                            thinking_started = True
                            console.print(f"[bold magenta]<think>[/]")
                            console.print(f"[dim cyan]{tok}[/]", end="", flush=True)
                            tokens_out.append(tok)
                            continue
                        console.print(f"[dim cyan]{tok}[/]", end="", flush=True)
                        tokens_out.append(tok)
                except Exception as e:
                    console.print(f"\n[bold red][error][/] Chat failed: {e}")
                    break
            else:
                try:
                    for chunk in self._llm.create_chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        stream=True,
                    ):
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            print(content, end="", flush=True)
                            tokens_out.append(content)
                except Exception as e:
                    console.print(f"\n[bold red][error][/] Chat failed: {e}")
                    break

            elapsed = time.perf_counter() - t0
            response = "".join(tokens_out)
            messages.append({"role": "assistant", "content": response})

            n_tok = len(tokens_out)
            tps = n_tok / elapsed if elapsed > 0 else 0
            console.print(f"\n\n[dim][{n_tok} tokens | {tps:.1f} tok/s][/]\n")

    def benchmark(
        self,
        prompt: str,
        max_tokens: int = 256,
        runs: int = 3,
    ) -> list[dict]:
        results = []
        for i in range(runs):
            t0 = time.perf_counter()
            raw = self._llm(
                prompt=prompt, max_tokens=max_tokens, temperature=0.0, echo=False
            )
            elapsed = time.perf_counter() - t0
            n = raw.get("usage", {}).get("completion_tokens", max_tokens)
            ts = n / elapsed if elapsed > 0 else 0
            results.append(
                {
                    "run": i + 1,
                    "tokens": n,
                    "tok_per_sec": round(ts, 1),
                    "elapsed": round(elapsed, 2),
                }
            )
        return results

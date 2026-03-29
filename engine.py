"""
engine.py — Core inference engine for GGUF models.

Handles:
  - Model registry (models.json)
  - Native GGUF metadata parsing (Chat templates, EOG tokens, n_ctx)
  - True multi-sequence parallel inference via C API bindings
  - Auto GPU offload detection
"""

from __future__ import annotations

import json
import os
import sys
import time
import multiprocessing
from pathlib import Path
from typing import Iterator
from contextlib import contextmanager

import warnings
import re

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import pynvml
    _HAS_PYNVML = True
except ImportError:
    _HAS_PYNVML = False

@contextmanager
def suppress_c_stderr():
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
    path_str = reg[alias]["path"]
    path = Path(path_str)
    
    if not path.exists():
        sys.exit(f"[error] Model path missing: {path}")

    # If it's a directory, find the first .gguf file inside
    if path.is_dir():
        ggufs = list(path.glob("*.gguf"))
        if not ggufs:
            sys.exit(f"[error] No .gguf file found in directory: {path}")
        return str(ggufs[0])
    
    return str(path)


def list_models() -> dict:
    return _load_registry()


def get_model_capabilities(alias: str) -> dict:
    """Read native GGUF metadata directly to determine limits."""
    from llama_cpp import Llama
    
    path = resolve_model(alias)
    p = Path(path)
    meta = _load_registry().get(alias, {})
    
    caps = {
        "alias": alias,
        "source": meta.get("source", "local"),
        "path": str(p),
        "exists": p.exists(),
        "size": f"{p.stat().st_size / 1e9:.2f} GB" if p.exists() else "0 GB",
        "total_context_length": 4096,
    }
    
    if p.exists():
        try:
            with suppress_c_stderr():
                temp_llm = Llama(model_path=path, vocab_only=True, verbose=False)
                
                # Dynamic key search: standard GGUF metadata for max context
                found_ctx = None
                for k, v in temp_llm.metadata.items():
                    if k.endswith(".context_length") and not k.endswith(".original_context_length"):
                        try:
                            found_ctx = int(v)
                            break
                        except (ValueError, TypeError):
                            continue
                
                if found_ctx:
                    caps["total_context_length"] = found_ctx
                del temp_llm
        except Exception:
            pass
            
    return caps


def get_model_config(alias: str) -> dict:
    reg = _load_registry()
    if alias not in reg:
        return {}
        
    model_dir = Path(reg[alias]["path"])
    params_path = model_dir / "params.json"
    
    config = {
        "configured_context_length": None,
        "max_input_tokens": "auto",
        "max_output_tokens": "auto",
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
    }
    
    if params_path.exists():
        try:
            with params_path.open() as f:
                data = json.load(f)
                config.update({k: data[k] for k in data if k in config or k in ["temperature", "top_p", "top_k"]})
        except Exception:
            pass
            
    return config


def set_model_config(alias: str, config: dict) -> None:
    reg = _load_registry()
    if alias not in reg:
        return
        
    model_dir = Path(reg[alias]["path"])
    model_dir.mkdir(parents=True, exist_ok=True)
    params_path = model_dir / "params.json"
    
    existing = {}
    if params_path.exists():
        try:
            with params_path.open() as f:
                existing = json.load(f)
        except Exception:
            pass
            
    existing.update(config)
    params_path.write_text(json.dumps(existing, indent=2))


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _detect_max_gpu_layers(model_path: str, alias: str) -> int:
    if not _HAS_PYNVML:
        return -1

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_vram_mb = info.free / 1024 / 1024
        pynvml.nvmlShutdown()

        available_mb = free_vram_mb - 1536
        if available_mb <= 0:
            print(f"[engine] Smart-Max: Low VRAM ({free_vram_mb:.0f}MB). Keeping on CPU.")
            return 0

        model_size_mb = Path(model_path).stat().st_size / 1024 / 1024
        
        if model_size_mb > 15000:
            total_layers = 80
        elif model_size_mb > 10000:
            total_layers = 62
        elif model_size_mb > 5000:
            total_layers = 35
        else:
            total_layers = 26

        eff_model_size_mb = model_size_mb * 1.15
        mb_per_layer = eff_model_size_mb / total_layers
        
        max_layers = int(available_mb / mb_per_layer)
        max_layers = max(0, min(max_layers, total_layers))
        
        if max_layers >= total_layers:
            print(f"[engine] Smart-Max: Fitting all {total_layers} layers in VRAM.")
            return -1
        
        print(f"[engine] Smart-Max: Detected {free_vram_mb:.0f}MB free. Offloading {max_layers}/{total_layers} layers.")
        return max_layers

    except Exception:
        return -1


def _detect_gpu_layers(model_path: str, alias: str) -> int:
    env = os.environ.get("LLM_GPU_LAYERS")
    if env is not None:
        if env.lower() == "max":
            return _detect_max_gpu_layers(model_path, alias)
        return int(env)
    return _detect_max_gpu_layers(model_path, alias)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

_DEFAULT_N_CTX    = int(os.environ.get("LLM_N_CTX",    "4096"))
_DEFAULT_N_BATCH  = int(os.environ.get("LLM_N_BATCH",  "512"))
_DEFAULT_N_THREADS = int(os.environ.get("LLM_N_THREADS", str(min(multiprocessing.cpu_count(), 8))))
# Used for prefill (prompt processing). Using all cores yields the fastest prompt ingestion.
_MAX_THREADS_BATCH = multiprocessing.cpu_count()


class Engine:
    """
    Thin wrapper around the GGUF runtime that heavily relies on the native
    metadata embedded in the .gguf file itself for 100% accuracy.
    """

    def __init__(
        self,
        alias: str,
        n_ctx: int | None = None,
        n_batch: int = _DEFAULT_N_BATCH,
        n_threads: int = _DEFAULT_N_THREADS,
        n_gpu_layers: int | str | None = None,
    ) -> None:
        from llama_cpp import Llama

        model_path = str(Path(resolve_model(alias)).resolve())
        model_size_mb = Path(model_path).stat().st_size / 1024 / 1024

        config = get_model_config(alias)
        caps = get_model_capabilities(alias)
        
        if n_ctx is None:
            n_ctx = config.get("configured_context_length") or _DEFAULT_N_CTX
        if caps.get("total_context_length"):
            n_ctx = min(n_ctx, caps["total_context_length"])

        if n_batch is None:
            n_batch = _DEFAULT_N_BATCH
        if n_threads is None:
            n_threads = _DEFAULT_N_THREADS

        self._config = config
        self._caps = caps
        self._alias = alias
        self._n_ctx = n_ctx

        if n_gpu_layers == "max":
             gpu_layers = _detect_max_gpu_layers(model_path, alias)
        elif n_gpu_layers is not None:
             gpu_layers = int(n_gpu_layers)
        else:
             gpu_layers = _detect_gpu_layers(model_path, alias)

        print(f"[engine] Loading '{alias}' …")
        print(f"         path       : {model_path}")
        print(f"         n_ctx      : {n_ctx}")
        print(f"         n_batch    : {n_batch}")
        print(f"         n_threads  : {n_threads} (Eval) / {_MAX_THREADS_BATCH} (Prefill)")
        print(f"         n_gpu_layers: {gpu_layers}")

        # Safety patch for some environments
        import llama_cpp._internals
        original_del = llama_cpp._internals.LlamaModel.__del__
        def safe_del(obj):
            try:
                original_del(obj)
            except Exception:
                pass
        llama_cpp._internals.LlamaModel.__del__ = safe_del

        self._llm = None
        attempts = 0
        max_attempts = 20
        
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
                        flash_attn=True,
                    )
            except (ValueError, RuntimeError) as e:
                attempts += 1
                if gpu_layers <= 0:
                    raise RuntimeError(f"Engine failed to load even on CPU: {e}")
                
                step = 1
                old_layers = gpu_layers if gpu_layers != -1 else "all"
                
                if gpu_layers == -1:
                    if model_size_mb > 15000:
                        gpu_layers = 80
                    elif model_size_mb > 10000:
                        gpu_layers = 62
                    elif model_size_mb > 5000:
                        gpu_layers = 35
                    else:
                        gpu_layers = 26
                
                gpu_layers = max(0, gpu_layers - step)
                print(f"[warn] VRAM limit reached at {old_layers} layers. Retrying with {gpu_layers}...")
                time.sleep(0.3)

        if not self._llm:
            raise RuntimeError(f"Failed to load engine for '{alias}' after {max_attempts} attempts.")
            
        self._alias = alias
        
        if self._llm.metadata.get("tokenizer.chat_template"):
            print("[engine] Native Chat Template detected in GGUF metadata.")
        else:
            print("[warn] No native chat template found in GGUF. Using fallback formatting.")
            
        print("[engine] Ready.\n")

    def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        """
        Extracts the native Jinja2 template embedded directly inside the GGUF file
        and evaluates it perfectly without relying on external configuration files.
        """
        template = self._llm.metadata.get("tokenizer.chat_template")
        
        if template:
            try:
                import jinja2
                env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
                
                # Setup builtins expected by many GGUF Jinja2 templates
                def _raise_exception(msg):
                    raise ValueError(msg)
                env.globals['raise_exception'] = _raise_exception
                
                # Fetch exact tokens from the model vocab
                bos_token = self._llm.detokenize([self._llm.token_bos()]).decode("utf-8") if self._llm.token_bos() != -1 else ""
                eos_token = self._llm.detokenize([self._llm.token_eos()]).decode("utf-8") if self._llm.token_eos() != -1 else ""
                
                jinja_tmpl = env.from_string(template)
                prompt = jinja_tmpl.render(
                    messages=messages, 
                    bos_token=bos_token, 
                    eos_token=eos_token, 
                    add_generation_prompt=add_generation_prompt
                )
                return prompt
            except Exception as e:
                print(f"[warn] Failed to evaluate native Jinja template: {e}. Falling back.")
        
        # Absolute fallback if the GGUF literally lacks any template
        prompt = ""
        for m in messages:
            role = m["role"].upper()
            content = m["content"]
            prompt += f"[{role}]: {content}\n"
        if add_generation_prompt:
            prompt += "[ASSISTANT]:"
        return prompt

    # ------------------------------------------------------------------
    # Single-prompt & Stream Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stream: bool = False,
        chat_format: bool = False,
    ) -> str | Iterator[str]:
        
        if temperature is None:
            temperature = float(self._config.get("temperature", 0.7))
        if top_p is None:
            top_p = float(self._config.get("top_p", 0.9))
        if top_k is None:
            top_k = int(self._config.get("top_k", 40))

        if max_tokens is None:
            saved = self._config.get("max_output_tokens")
            if saved == "auto" or saved is None:
                input_toks = self._llm.tokenize(prompt.encode("utf-8"))
                max_tokens = max(128, self._n_ctx - len(input_toks) - 10)
            else:
                max_tokens = int(saved)

        # Chat models heavily benefit from the native create_chat_completion API.
        if chat_format:
            messages = [{"role": "user", "content": prompt}]
            
            if stream:
                def _gen():
                    for chunk in self._llm.create_chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        stream=True
                    ):
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                return _gen()
            
            result = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            return result["choices"][0]["message"]["content"]

        # Raw Completion mode
        kwargs = dict(
            prompt=prompt,
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

        result = self._llm(**kwargs)
        return result["choices"][0]["text"]

    # ------------------------------------------------------------------
    # Parallel Batch Generation (C-Level Bindings)
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
    ) -> list[dict]:
        
        if temperature is None:
            temperature = float(self._config.get("temperature", 0.7))
        if top_p is None:
            top_p = float(self._config.get("top_p", 0.9))
        if top_k is None:
            top_k = int(self._config.get("top_k", 0))

        if max_tokens is None:
            saved = self._config.get("max_output_tokens")
            if saved == "auto":
                max_tokens = min(2048, self._n_ctx // 4)
            else:
                max_tokens = int(saved or 2048)
                
        all_results: list[dict] = []
        total_tokens = 0
        wall_start = time.perf_counter()
        done_count = 0

        # --- Evaluate templates and Tokenize ---
        tokenized = []
        for p in prompts:
            actual_prompt = p
            if chat_format:
                actual_prompt = self.apply_chat_template([{"role": "user", "content": p}])
            toks = self._llm.tokenize(actual_prompt.encode("utf-8"), add_bos=True)
            tokenized.append(toks)

        # --- Sub-batch Context Calculation ---
        if parallel > 0:
            sub_batches: list[list[int]] = []
            for start in range(0, len(prompts), parallel):
                sub_batches.append(list(range(start, min(start + parallel, len(prompts)))))
        else:
            sub_batches = []
            current_batch: list[int] = []
            current_ctx_usage = 0
            n_ctx = self._llm._n_ctx

            for idx, toks in enumerate(tokenized):
                seq_budget = len(toks) + max_tokens
                if seq_budget > n_ctx:
                    if current_batch:
                        sub_batches.append(current_batch)
                    sub_batches.append([idx])
                    current_batch = []
                    current_ctx_usage = 0
                    continue

                if current_ctx_usage + seq_budget > n_ctx and current_batch:
                    sub_batches.append(current_batch)
                    current_batch = []
                    current_ctx_usage = 0

                current_batch.append(idx)
                current_ctx_usage += seq_budget

            if current_batch:
                sub_batches.append(current_batch)

        n_sub = len(sub_batches)
        if n_sub > 1:
            print(f"[batch] Split into {n_sub} sub-batches (parallel={'auto' if parallel == 0 else parallel})")

        for sb_i, sb_indices in enumerate(sub_batches):
            sb_prompts = [prompts[i] for i in sb_indices]
            sb_tokens  = [tokenized[i] for i in sb_indices]
            n_seq      = len(sb_indices)

            if n_sub > 1:
                print(f"\n[batch] Sub-batch {sb_i+1}/{n_sub}: {n_seq} prompts in parallel")

            sb_results = self._parallel_decode(
                sb_prompts, sb_tokens, max_tokens, temperature, top_p,
                start_idx=done_count, total=len(prompts),
                top_k=top_k,
            )

            for r in sb_results:
                total_tokens += r["tokens"]
            all_results.extend(sb_results)
            done_count += n_seq

        wall = time.perf_counter() - wall_start
        agg = total_tokens / wall if wall > 0 else 0.0
        print(f"\n[batch] {total_tokens} total tokens in {wall:.2f}s → {agg:.1f} tok/s aggregate")
        return all_results

    def _parallel_decode(
        self,
        prompts: list[str],
        prompt_tokens: list[list[int]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        start_idx: int = 0,
        total: int = 0,
        top_k: int = 0,
    ) -> list[dict]:
        import llama_cpp.llama_cpp as ll

        llm = self._llm
        n_batch = llm.n_batch
        n_seq = len(prompts)

        required_ctx = sum(len(t) + max_tokens for t in prompt_tokens)
        batch_n_ctx = max(required_ctx, llm.context_params.n_ctx)
        
        ctx_params = ll.llama_context_default_params()
        ctx_params.n_ctx = batch_n_ctx
        ctx_params.n_batch = max(llm.context_params.n_batch, n_seq)
        ctx_params.n_seq_max = n_seq
        ctx_params.n_threads = llm.context_params.n_threads
        ctx_params.n_threads_batch = _MAX_THREADS_BATCH
        
        for attr in ['flash_attn', 'flash_attn_type', 'attention_type', 'offload_kqv']:
            if hasattr(llm.context_params, attr):
                setattr(ctx_params, attr, getattr(llm.context_params, attr))

        batch_ctx = ll.llama_init_from_model(llm.model, ctx_params)
        if batch_ctx is None:
            raise RuntimeError(f"Failed to create batch context (n_seq={n_seq}, n_ctx={batch_n_ctx}). VRAM exhausted.")

        vocab = llm._model.vocab

        samplers = []
        for seq_id in range(n_seq):
            smplr = llm._init_sampler(
                temp=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=0.05,
                repeat_penalty=1.1,
                frequency_penalty=0.1,
                presence_penalty=0.1,
            )
            samplers.append(smplr)

        generated: list[list[int]] = [[] for _ in range(n_seq)]
        seq_pos: list[int] = [0] * n_seq
        finished: list[bool] = [False] * n_seq
        t0 = time.perf_counter()

        # Phase 1: Prefill all except the last token
        non_last_prefill = []
        last_tokens = []
        for seq_id, toks in enumerate(prompt_tokens):
            for i, tok in enumerate(toks[:-1]):
                non_last_prefill.append((tok, i, seq_id))
            last_tokens.append((toks[-1], len(toks)-1, seq_id))
            seq_pos[seq_id] = len(toks)

        for chunk_start in range(0, len(non_last_prefill), n_batch):
            chunk = non_last_prefill[chunk_start : chunk_start + n_batch]
            batch = ll.llama_batch_init(len(chunk), 0, 1)
            batch.n_tokens = len(chunk)
            for j, (tok, pos, seq_id) in enumerate(chunk):
                batch.token[j] = tok
                batch.pos[j] = pos
                batch.n_seq_id[j] = 1
                batch.seq_id[j][0] = seq_id
                batch.logits[j] = False
            ll.llama_decode(batch_ctx, batch)
            ll.llama_batch_free(batch)

        # Phase 1.5: Final Prefill Token and First Output
        batch = ll.llama_batch_init(n_seq, 0, 1)
        batch.n_tokens = n_seq
        for j, (tok, pos, seq_id) in enumerate(last_tokens):
            batch.token[j] = tok
            batch.pos[j] = pos
            batch.n_seq_id[j] = 1
            batch.seq_id[j][0] = seq_id
            batch.logits[j] = True
        
        ll.llama_decode(batch_ctx, batch)
        
        for seq_i in range(n_seq):
            seq_id = last_tokens[seq_i][2] 
            token = ll.llama_sampler_sample(samplers[seq_id].sampler, batch_ctx, seq_i)
            ll.llama_sampler_accept(samplers[seq_id].sampler, token)
            
            # Use native C-level check for End-Of-Generation (EOS, EOT, etc.)
            if ll.llama_vocab_is_eog(vocab, token):
                finished[seq_id] = True
            else:
                generated[seq_id].append(token)
        ll.llama_batch_free(batch)

        # Phase 2: Autoregressive decoding parallel step
        for step in range(1, max_tokens):
            active = [s for s in range(n_seq) if not finished[s] and len(generated[s]) < max_tokens]
            if not active:
                break

            batch = ll.llama_batch_init(len(active), 0, 1)
            batch.n_tokens = len(active)
            for j, seq_id in enumerate(active):
                batch.token[j] = generated[seq_id][-1]
                batch.pos[j] = seq_pos[seq_id]
                batch.n_seq_id[j] = 1
                batch.seq_id[j][0] = seq_id
                batch.logits[j] = True
                seq_pos[seq_id] += 1

            if ll.llama_decode(batch_ctx, batch) != 0:
                ll.llama_batch_free(batch)
                break

            for j, seq_id in enumerate(active):
                token = ll.llama_sampler_sample(samplers[seq_id].sampler, batch_ctx, j)
                ll.llama_sampler_accept(samplers[seq_id].sampler, token)
                
                # Lightning-fast native stop token validation (no string ops required)
                if ll.llama_vocab_is_eog(vocab, token):
                    finished[seq_id] = True
                else:
                    generated[seq_id].append(token)

            ll.llama_batch_free(batch)

        elapsed = time.perf_counter() - t0

        results = []
        total_gen = sum(len(g) for g in generated)
        tok_per_sec = total_gen / elapsed if elapsed > 0 else 0.0

        for seq_id in range(n_seq):
            text = llm.detokenize(generated[seq_id]).decode("utf-8", errors="replace")
            n_tok = len(generated[seq_id])
            results.append({
                "prompt": prompts[seq_id],
                "output": text,
                "tokens": n_tok,
                "elapsed": round(elapsed, 3),
                "tok_per_sec": round(tok_per_sec, 1),
            })
            global_idx = start_idx + seq_id + 1
            print(f"  [{global_idx}/{total}] {n_tok} tokens  (parallel batch)")

        print(f"  → batch decoded {n_seq} prompts in {elapsed:.2f}s — "
              f"{total_gen} tokens total — {tok_per_sec:.1f} tok/s effective")

        for s in samplers:
            s.close()
        ll.llama_free(batch_ctx)

        return results

    # ------------------------------------------------------------------
    # Interactive chat (REPL)
    # ------------------------------------------------------------------

    def chat(
        self,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> None:
        
        if temperature is None:
            temperature = float(self._config.get("temperature", 0.7))
        if top_p is None:
            top_p = float(self._config.get("top_p", 0.9))
        if top_k is None:
            top_k = int(self._config.get("top_k", 40))

        if max_tokens is None:
            saved = self._config.get("max_output_tokens")
            max_tokens = min(4096, self._n_ctx // 2) if saved == "auto" else int(saved or 2048)

        messages = [{"role": "system", "content": system_prompt}]
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

            messages.append({"role": "user", "content": user_input})
            tokens_out = []
            print("\033[34mAssistant:\033[0m ", end="", flush=True)
            t0 = time.perf_counter()

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
                    if "content" in delta:
                        tok = delta["content"]
                        print(tok, end="", flush=True)
                        tokens_out.append(tok)
            except Exception as e:
                print(f"\n[error] Chat completion failed: {e}")
                break

            elapsed = time.perf_counter() - t0
            response = "".join(tokens_out)
            messages.append({"role": "assistant", "content": response})
            
            n_tok = len(tokens_out)
            tps = n_tok / elapsed if elapsed > 0 else 0
            print(f"\n\033[90m[{n_tok} tokens | {tps:.1f} tok/s]\033[0m\n")
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


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _detect_gpu_layers() -> int:
    """
    Returns -1 to strictly enforce maximum GPU offloading for all layers,
    as requested by the user.
    """
    env = os.environ.get("LLM_GPU_LAYERS")
    if env is not None:
        return int(env)

    return -1


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
        from llama_cpp import Llama, llama_chat_format  # lazy import — lets the CLI load fast

        model_path = str(Path(resolve_model(alias)).resolve())
        model_dir = Path(model_path).parent
        gpu_layers = n_gpu_layers if n_gpu_layers is not None else _detect_gpu_layers()

        print(f"[engine] Loading '{alias}' …")
        print(f"         path       : {model_path}")
        print(f"         n_ctx      : {n_ctx}")
        print(f"         n_batch    : {n_batch}")
        print(f"         n_threads  : {n_threads}")
        print(f"         n_gpu_layers: {gpu_layers}")

        # --- Detect Chat Template & Params -------------------------
        chat_handler = None
        self._custom_stop = []

        # 1. Try JSON configuration files (Standard HF)
        json_candidates = ["tokenizer_config.json", "generation_config.json", "params.json"]
        for jc in json_candidates:
            jp = model_dir / jc
            if jp.exists():
                try:
                    with jp.open() as f:
                        data = json.load(f)
                    
                    # Look for chat template
                    if "chat_template" in data and not chat_handler:
                        print(f"[engine] Using chat template from {jc}")
                        chat_handler = llama_chat_format.hf_tokenizer_config_to_chat_completion_handler(data)
                    
                    # Look for stop tokens
                    if "stop" in data and not self._custom_stop:
                        self._custom_stop = data["stop"]
                        print(f"[engine] Loaded custom stop tokens from {jc}")
                except Exception as e:
                    print(f"[warn] Error parsing {jc}: {e}")

        # 2. Try Raw Template files (template, template.jinja, etc.)
        if not chat_handler:
            raw_candidates = ["template", "template.jinja", "template.jinja2", "chat_template"]
            for rc in raw_candidates:
                rp = model_dir / rc
                if rp.exists():
                    try:
                        t_str = rp.read_text().strip()
                        
                        # Robust check: if it looks like a Go template (Ollama/Dot-syntax), 
                        # Jinja2 will fail. Convert or fallback.
                        if ".Role" in t_str and ".Content" in t_str:
                            print(f"[engine] Raw template in '{rc}' is Go-style. Converting to Jinja2...")
                            # Simple conversion for standard ChatML-like Go templates
                            if "<|im_start|>" in t_str:
                                t_str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
                        
                        print(f"[engine] Using raw chat template from {rc}")
                        # Some versions of llama-cpp-python assert that common tokens exist in the dict
                        dummy_config = {
                            "chat_template": t_str,
                            "bos_token": "<|endoftext|>",
                            "eos_token": "<|endoftext|>",
                        }
                        chat_handler = llama_chat_format.hf_tokenizer_config_to_chat_completion_handler(dummy_config)
                        break
                    except Exception as e:
                        print(f"[warn] Error reading {rc}: {repr(e)}")

        # 3. Try Raw Params (non-standard 'params' file)
        pp = model_dir / "params"
        if pp.exists() and not self._custom_stop:
            try:
                # Some 'params' are JSON, some are key-value
                content = pp.read_text()
                if content.strip().startswith("{"):
                    p_data = json.loads(content)
                    self._custom_stop = p_data.get("stop", [])
                    print("[engine] Loaded stop tokens from 'params' (JSON)")
            except Exception:
                pass

        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            n_gpu_layers=gpu_layers,
            verbose=False,
            logits_all=False,
            use_mmap=True,
            chat_handler=chat_handler,
        )
        self._alias = alias
        
        # Detect chat template
        self.chat_format = self._llm.chat_format
        template = self._llm.metadata.get("tokenizer.chat_template")
        
        if chat_handler:
            print("[engine] Using external HF chat handler.")
        elif template:
            print(f"[engine] Chat template detected in GGUF: {self.chat_format}")
        else:
            print("[engine] No chat template found, using fallback.")
        
        print("[engine] Ready.\n")

    def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        """
        Formats a list of messages into a single prompt string using the model's 
        internal chat template (Jinja2) or a fallback format.
        """
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        
        template = self._llm.metadata.get("tokenizer.chat_template")
        if template:
            try:
                # Use the model's own BOS/EOS tokens
                bos = self._llm.detokenize([self._llm.token_bos()]).decode("utf-8", errors="ignore")
                eos = self._llm.detokenize([self._llm.token_eos()]).decode("utf-8", errors="ignore")
                
                formatter = Jinja2ChatFormatter(template, eos, bos, add_generation_prompt=add_generation_prompt)
                return formatter(messages=messages).prompt
            except Exception as e:
                print(f"[warn] Failed to apply Jinja2 template: {e}. Falling back.")
        
        # Fallback format (the "others" template mentioned by user)
        prompt = ""
        for m in messages:
            role = m["role"].upper()
            content = m["content"]
            prompt += f"[{role}]: {content}\n"
        if add_generation_prompt:
            prompt += "[ASSISTANT]:"
        return prompt

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
        chat_format: bool = False,
    ) -> str | Iterator[str]:
        """
        Generate text for a single prompt.

        Args:
            prompt:     Input text.
            max_tokens: Maximum tokens to generate.
            temperature / top_p: Sampling parameters.
            stream:     If True, yields tokens as a generator.
            chat_format: If True, wraps the prompt in the model's chat template.

        Returns:
            Complete string (stream=False) or token iterator (stream=True).
        """
        if chat_format:
            # Wrap as a simple user message
            messages = [{"role": "user", "content": prompt}]
            
            # For chat-formatted prompts, we also need to handle stop tokens
            # We can use create_chat_completion which handles this automatically
            stop = self._custom_stop
            if stream:
                def _gen():
                    for chunk in self._llm.create_chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stream=True,
                        stop=stop,
                    ):
                        delta = chunk["choices"][0]["delta"]
                        if "content" in delta:
                            yield delta["content"]
                return _gen()
            
            result = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False,
                stop=stop,
            )
            return result["choices"][0]["message"]["content"]

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
    # Batch generation (true parallel via llama.cpp multi-sequence batch)
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        parallel: int = 0,
        chat_format: bool = False,
    ) -> list[dict]:
        """
        Run inference over a list of prompts **in parallel** using llama.cpp's
        native multi-sequence batch API.

        Args:
            parallel: Max prompts per sub-batch.
            chat_format: If True, wraps each prompt in the model's chat template.
                      0 = auto (fit as many as the engine's n_ctx allows).
                      >0 = fixed sub-batch size; the batch context is auto-
                            scaled so all prompts fit regardless of n_ctx.

        Returns a list of dicts:
          {"prompt": str, "output": str, "tokens": int, "elapsed": float, "tok_per_sec": float}
        """
        all_results: list[dict] = []
        total_tokens = 0
        wall_start = time.perf_counter()
        done_count = 0

        # --- Tokenize all prompts up-front ----------------------------
        tokenized = []
        for p in prompts:
            actual_prompt = p
            if chat_format:
                actual_prompt = self.apply_chat_template([{"role": "user", "content": p}])
            
            toks = self._llm.tokenize(actual_prompt.encode("utf-8"), add_bos=True)
            tokenized.append(toks)

        # --- Build sub-batches ----------------------------------------
        if parallel > 0:
            # Force the requested parallelism. 
            # We skip all safety split checks to fulfill the high-throughput goal.
            sub_batches: list[list[int]] = []
            for start in range(0, len(prompts), parallel):
                sub_batches.append(list(range(start, min(start + parallel, len(prompts)))))
        else:
            # Auto: fit as many as the engine's n_ctx allows
            sub_batches = []
            current_batch: list[int] = []
            current_ctx_usage = 0
            n_ctx = self._llm._n_ctx # Need n_ctx for auto-splitting

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
            print(f"[batch] Split into {n_sub} sub-batches "
                  f"(parallel={'auto' if parallel == 0 else parallel})")

        # --- Process each sub-batch in true parallel ------------------
        for sb_i, sb_indices in enumerate(sub_batches):
            sb_prompts = [prompts[i] for i in sb_indices]
            sb_tokens  = [tokenized[i] for i in sb_indices]
            n_seq      = len(sb_indices)

            if n_sub > 1:
                print(f"\n[batch] Sub-batch {sb_i+1}/{n_sub}: {n_seq} prompts in parallel")

            sb_results = self._parallel_decode(
                sb_prompts, sb_tokens, max_tokens, temperature, top_p,
                start_idx=done_count, total=len(prompts),
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
    ) -> list[dict]:
        """
        Decode multiple prompts in parallel using the low-level llama.cpp
        batch API.  All sequences share one llama_decode() call per step.
        """
        import llama_cpp.llama_cpp as ll

        llm = self._llm
        n_batch = llm.n_batch  # max tokens per llama_decode call
        n_seq = len(prompts)

        # ---- Dynamic Context Allocation ------------------------------
        # To fulfill the high-throughput goal, we automatically calculate
        # the exact context needed to fit these parallel sequences.
        required_ctx = sum(len(t) + max_tokens for t in prompt_tokens)
        
        # We use the higher of the requested context or the required context
        batch_n_ctx = max(required_ctx, llm.context_params.n_ctx)
        
        ctx_params = ll.llama_context_default_params()
        ctx_params.n_ctx = batch_n_ctx
        ctx_params.n_batch = max(llm.context_params.n_batch, n_seq)
        ctx_params.n_seq_max = n_seq
        ctx_params.n_threads = llm.context_params.n_threads
        ctx_params.n_threads_batch = llm.context_params.n_threads_batch
        
        for attr in ['flash_attn', 'flash_attn_type', 'attention_type', 'offload_kqv']:
            if hasattr(llm.context_params, attr):
                setattr(ctx_params, attr, getattr(llm.context_params, attr))

        batch_ctx = ll.llama_init_from_model(llm.model, ctx_params)
        if batch_ctx is None:
            raise RuntimeError(
                f"Failed to create batch context (n_seq={n_seq}, "
                f"n_ctx={batch_n_ctx}). You may be out of VRAM."
            )

        vocab = llm._model.vocab  # raw llama_vocab_p

        # ---- Create per-sequence samplers ----------------------------
        # Each sequence needs its own sampler with:
        #   - repetition penalty to avoid loops
        #   - unique seed for diversity across parallel sequences
        samplers = []
        for seq_id in range(n_seq):
            smplr = llm._init_sampler(
                temp=temperature,
                top_p=top_p,
                top_k=40,
                min_p=0.05,
                repeat_penalty=1.1,
                frequency_penalty=0.1,
                presence_penalty=0.1,
            )
            samplers.append(smplr)

        # ---- State tracking per sequence -----------------------------
        generated: list[list[int]] = [[] for _ in range(n_seq)]  # generated token ids
        seq_pos: list[int] = [0] * n_seq   # current position in KV cache per seq
        finished: list[bool] = [False] * n_seq
        t0 = time.perf_counter()

        # ---- Phase 1: Prefill — process all tokens except the last one ---
        # We process N-1 tokens for all sequences first (logits=False).
        non_last_prefill = []
        last_tokens = []
        for seq_id, toks in enumerate(prompt_tokens):
            for i, tok in enumerate(toks[:-1]):
                non_last_prefill.append((tok, i, seq_id))
            last_tokens.append((toks[-1], len(toks)-1, seq_id))
            seq_pos[seq_id] = len(toks)

        # Feed non-last tokens in chunks of n_batch
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
            rc = ll.llama_decode(batch_ctx, batch)
            if rc != 0:
                print(f"[warn] llama_decode (prefill) returned {rc}")
            ll.llama_batch_free(batch)

        # ---- Phase 1.5: Final Prefill Token — Sample first token for each ---
        # Decode the last token of every sequence together.
        # This guarantees logits are available for everyone in index 0..n_seq-1.
        batch = ll.llama_batch_init(n_seq, 0, 1)
        batch.n_tokens = n_seq
        for j, (tok, pos, seq_id) in enumerate(last_tokens):
            batch.token[j] = tok
            batch.pos[j] = pos
            batch.n_seq_id[j] = 1
            batch.seq_id[j][0] = seq_id
            batch.logits[j] = True
        
        rc = ll.llama_decode(batch_ctx, batch)
        if rc != 0:
            print(f"[warn] llama_decode (last-prefill) returned {rc}")
        
        # Sample the very first generated token for all sequences
        for seq_i in range(n_seq):
            # Index 0..n_seq-1 in THIS batch corresponds to the j-th seq in order
            seq_id = last_tokens[seq_i][2] 
            token = ll.llama_sampler_sample(samplers[seq_id].sampler, batch_ctx, seq_i)
            ll.llama_sampler_accept(samplers[seq_id].sampler, token)
            if ll.llama_vocab_is_eog(vocab, token):
                finished[seq_id] = True
            else:
                generated[seq_id].append(token)
        ll.llama_batch_free(batch)

        # ---- Phase 2: Autoregressive decode — all seqs in parallel ---
        for step in range(1, max_tokens):
            # Collect active sequences
            active = [s for s in range(n_seq) if not finished[s] and len(generated[s]) < max_tokens]
            if not active:
                break

            # Build batch: one token per active sequence
            batch = ll.llama_batch_init(len(active), 0, 1)
            batch.n_tokens = len(active)
            for j, seq_id in enumerate(active):
                last_token = generated[seq_id][-1]
                pos = seq_pos[seq_id]
                batch.token[j] = last_token
                batch.pos[j] = pos
                batch.n_seq_id[j] = 1
                batch.seq_id[j][0] = seq_id
                batch.logits[j] = True   # need logits for all active seqs
                seq_pos[seq_id] = pos + 1

            rc = ll.llama_decode(batch_ctx, batch)
            if rc != 0:
                print(f"[warn] llama_decode (step {step}) returned {rc}")
                ll.llama_batch_free(batch)
                break

            # Sample one token per active sequence
            for j, seq_id in enumerate(active):
                token = ll.llama_sampler_sample(samplers[seq_id].sampler, batch_ctx, j)
                ll.llama_sampler_accept(samplers[seq_id].sampler, token)
                
                is_eog = ll.llama_vocab_is_eog(vocab, token)
                
                # Check for custom stop sequences
                is_custom_stop = False
                if self._custom_stop:
                    # Detokenize to check recently generated text
                    text = llm.detokenize(generated[seq_id] + [token]).decode("utf-8", errors="ignore")
                    for s in self._custom_stop:
                        if text.endswith(s):
                            is_custom_stop = True
                            break
                
                if is_eog or is_custom_stop:
                    finished[seq_id] = True
                else:
                    generated[seq_id].append(token)

            ll.llama_batch_free(batch)

        elapsed = time.perf_counter() - t0

        # ---- Build results -------------------------------------------
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

        # Cleanup samplers and batch context
        for s in samplers:
            s.close()
        ll.llama_free(batch_ctx)

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
        Simple REPL chat loop using the model's internal chat template.
        """
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

            # Use create_chat_completion to use the model's own template automatically
            try:
                for chunk in self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    stream=True,
                    stop=self._custom_stop,
                ):
                    delta = chunk["choices"][0]["delta"]
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

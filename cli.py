"""
cli.py — Command-line interface for the GGUF inference engine.

Usage:
  python cli.py load <hf_repo_or_url> --alias <name> [--file <filename>]
  python cli.py run  <alias> --prompt "text" [--max-tokens N] [--temp F] [--stream]
  python cli.py batch <alias> --file prompts.txt [--max-tokens N]
  python cli.py chat  <alias> [--system "You are..."] [--max-tokens N]
  python cli.py list
  python cli.py info <alias>
  python cli.py bench <alias> [--prompt "text"] [--runs N]

Environment variables for performance tuning:
  LLM_GPU_LAYERS  — override GPU layer count (0 = CPU-only)
  LLM_N_CTX      — context window size          (default: 4096)
  LLM_N_BATCH    — token batch size             (default: 512)
  LLM_N_THREADS  — CPU thread count             (default: min(cpus, 8))
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Subcommand: load
# ---------------------------------------------------------------------------

def cmd_load(args: argparse.Namespace) -> None:
    """Download a GGUF model from HuggingFace (or a direct URL) and register it."""
    from engine import MODELS_DIR, register_model

    source: str = args.source
    alias: str = args.alias
    filename: str = args.file  # optional explicit filename within the repo

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Destination is a directory named after the alias
    dest_dir = MODELS_DIR / alias

    # ---- HuggingFace repo  (hf://<org/repo> or plain <org/repo>) ----------
    if source.startswith("hf://") or (
        "/" in source and not source.startswith("http")
    ):
        repo_id = source.removeprefix("hf://")
        _download_hf(repo_id, dest_dir, filename)
        register_model(alias, str(dest_dir), source)

    # ---- Direct HTTPS URL -------------------------------------------------
    elif source.startswith("http://") or source.startswith("https://"):
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"{alias}.gguf"
        _download_url(source, dest_file)
        register_model(alias, str(dest_dir), source)

    # ---- Local file path --------------------------------------------------
    elif Path(source).exists():
        import shutil
        dest_dir.mkdir(parents=True, exist_ok=True)
        if Path(source).is_dir():
            print(f"[load] Copying local directory → {dest_dir}")
            shutil.copytree(source, dest_dir, dirs_exist_ok=True)
        else:
            print(f"[load] Copying local file → {dest_dir / Path(source).name}")
            shutil.copy2(source, dest_dir / Path(source).name)
        register_model(alias, str(dest_dir), source)

    else:
        sys.exit(f"[error] Cannot resolve source: '{source}'")

    print(f"[load] Done. Use:  python cli.py run {alias} --prompt \"Hello\" --chat")


def _download_hf(repo_id: str, dest_dir: Path, filename: str | None) -> Path:
    try:
        from huggingface_hub import snapshot_download, list_repo_files
    except ImportError:
        sys.exit("[error] huggingface_hub not installed. Run: pip install huggingface_hub")

    # 1. Identify all GGUF files in the repo
    all_files = list(list_repo_files(repo_id))
    gguf_files = [f for f in all_files if f.endswith(".gguf")]
    if not gguf_files:
        sys.exit(f"[error] No .gguf files found in {repo_id}")

    # 2. Select the specific GGUF file to download
    if not filename:
        # Preference: Q4_K_M -> Q4_K_S -> Q5_K_M -> Q8_0 -> first available
        preferred = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0"]
        filename = gguf_files[0]
        for tag in preferred:
            match = [f for f in gguf_files if tag in f]
            if match:
                filename = match[0]
                break
        print(f"[load] Auto-selected GGUF: {filename}")

    # 3. Download ONLY the target GGUF file.
    # The GGUF file natively contains all metadata (chat templates, configs, stops).
    other_ggufs = [f for f in gguf_files if f != filename]
    
    print(f"[load] Downloading {repo_id} (GGUF: {filename}) to {dest_dir} …")
    
    final_path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest_dir),
        allow_patterns=[filename],
        ignore_patterns=other_ggufs,
        local_dir_use_symlinks=False,
    )
    
    return Path(final_path) / filename


def _download_url(url: str, dest: Path) -> None:
    import urllib.request

    print(f"[load] Downloading {url} …")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            mb = downloaded / 1e6
            print(f"\r  {pct:.1f}%  {mb:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print(f"\n[load] Saved → {dest}  ({dest.stat().st_size / 1e9:.2f} GB)")


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    from engine import Engine, get_model_config

    # Load persistent config
    config_data = get_model_config(args.alias)
    
    # Priority: CLI flag > Persisted Config > Env/Default
    n_ctx = args.n_ctx or config_data.get("configured_context_length")
    
    engine = Engine(
        alias=args.alias,
        n_gpu_layers=args.gpu_layers,
        n_ctx=n_ctx,
        n_batch=args.n_batch,
        n_threads=args.n_threads,
    )

    prompt = args.prompt
    t0 = time.perf_counter()

    # Always stream for a better UX unless explicitly disabled
    stream = not args.no_stream 
    
    disp_max = args.max_tokens or config_data.get("max_output_tokens") or "auto"
    disp_temp = args.temp
    if disp_temp is None:
        disp_temp = config_data.get("temperature", 0.7)
    
    disp_top_p = args.top_p or config_data.get("top_p", 0.9)
    disp_top_k = args.top_k or config_data.get("top_k", 40)
    
    print(f"\033[1;36mGENERATION\033[0m")
    print(f"{'  Prompt':<12} : \"{prompt[:60]}{'...' if len(prompt)>60 else ''}\"")
    print(f"{'  Temp':<12} : {disp_temp:.2f}")
    print(f"{'  Top-P':<12} : {disp_top_p:.2f}")
    print(f"{'  Top-K':<12} : {disp_top_k}")
    if disp_max == "auto":
        inp_len = len(prompt.split()) * 1.3
        resolved_max = max(128, int(engine._n_ctx - inp_len - 20))
        print(f"{'  Max Tokens':<12} : auto ({resolved_max} available)")
    else:
        print(f"{'  Max Tokens':<12} : {disp_max}")
    print(f"{'  Mode':<12} : {'Streaming' if stream else 'Static'} ({'Chat' if args.chat else 'Completion'})")

    print("\n" + "─" * 50)
    tokens = 0
    max_tokens = args.max_tokens

    if stream:
        for tok in engine.generate(prompt, max_tokens=max_tokens,
                                   temperature=disp_temp,
                                   top_p=args.top_p,
                                   top_k=args.top_k,
                                   stream=True,
                                   chat_format=args.chat):
            print(tok, end="", flush=True)
            tokens += 1
            
        elapsed = time.perf_counter() - t0
        tps = tokens / elapsed if elapsed > 0 else 0
        print("\n" + "─" * 50)
        print(f"\033[90m[{tokens} tokens | {tps:.1f} tok/s | {elapsed:.2f}s]\033[0m")
    else:
        output = engine.generate(prompt, max_tokens=args.max_tokens,
                                 temperature=args.temp,
                                 top_p=args.top_p,
                                 top_k=args.top_k,
                                 stream=False,
                                 chat_format=args.chat)
        elapsed = time.perf_counter() - t0
        print(output)
        print("\n" + "─" * 50)
        print(f"\033[90m[{elapsed:.2f}s]\033[0m")
    print()


# ---------------------------------------------------------------------------
# Subcommand: batch
# ---------------------------------------------------------------------------

def cmd_batch(args: argparse.Namespace) -> None:
    from engine import Engine

    pfile = Path(args.file)
    if not pfile.exists():
        sys.exit(f"[error] Prompt file not found: {args.file}")

    prompts = [line.strip() for line in pfile.read_text().splitlines() if line.strip()]
    if not prompts:
        sys.exit("[error] No prompts found in file.")

    print(f"[batch] {len(prompts)} prompts loaded from {args.file}")

    from engine import Engine, get_model_config

    config_data = get_model_config(args.alias)
    n_ctx = args.n_ctx or config_data.get("configured_context_length")

    disp_max = args.max_tokens or config_data.get("max_output_tokens") or "auto"
    print(f"\033[1;36mGENERATION (Batch)\033[0m")
    print(f"{'  Source':<12} : {args.file} ({len(prompts)} prompts)")
    print(f"{'  Parallel':<12} : {args.parallel if args.parallel > 0 else 'auto'}")
    print(f"{'  Max Tokens':<12} : {disp_max}")
    print("\n" + "─" * 50)

    engine = Engine(
        alias=args.alias,
        n_gpu_layers=args.gpu_layers,
        n_ctx=n_ctx,
        n_batch=args.n_batch,
        n_threads=args.n_threads,
    )

    max_tokens = args.max_tokens

    results = engine.generate_batch(
        prompts,
        max_tokens=max_tokens,
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        parallel=args.parallel,
        chat_format=args.chat,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"[batch] Results written to {out_path}")
    else:
        print("\n--- Results ---")
        for r in results:
            print(f"\nPrompt : {r['prompt'][:80]}…")
            print(f"Output : {r['output'][:200]}…")
            print(f"Stats  : {r['tokens']} tokens, {r['tok_per_sec']} tok/s")


# ---------------------------------------------------------------------------
# Subcommand: chat
# ---------------------------------------------------------------------------

def cmd_chat(args: argparse.Namespace) -> None:
    from engine import Engine, get_model_config

    config_data = get_model_config(args.alias)
    n_ctx = args.n_ctx or config_data.get("configured_context_length")

    engine = Engine(
        alias=args.alias,
        n_gpu_layers=args.gpu_layers,
        n_ctx=n_ctx,
        n_batch=args.n_batch,
        n_threads=args.n_threads,
    )
    temp = args.temp
    if temp is None:
        temp = config_data.get("temperature", 0.7)

    engine.chat(
        system_prompt=args.system,
        max_tokens=args.max_tokens,
        temperature=temp,
        top_p=args.top_p,
        top_k=args.top_k,
    )


# ---------------------------------------------------------------------------
# Subcommand: list
# ---------------------------------------------------------------------------
def cmd_list(_args: argparse.Namespace) -> None:
    from engine import list_models

    reg = list_models()
    if not reg:
        print("[list] No models registered yet. Run: python cli.py load <repo> --alias <name>")
        return

    print(f"\n{'ALIAS':<20}  {'SIZE':>8}  {'SOURCE'}")
    print("-" * 70)
    for alias, meta in reg.items():
        p = Path(meta["path"])
        if not p.exists():
            size = "MISSING"
        elif p.is_dir():
            total_bytes = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            size = f"{total_bytes / 1e9:.2f}G"
        else:
            size = f"{p.stat().st_size / 1e9:.2f}G"
        print(f"{alias:<20}  {size:>8}  {meta['source']}")
    print()


# ---------------------------------------------------------------------------
# Subcommand: config
# ---------------------------------------------------------------------------

def cmd_config(args: argparse.Namespace) -> None:
    from engine import get_model_capabilities, get_model_config, set_model_config
    
    alias = args.alias
    caps = get_model_capabilities(alias)
    if not caps.get("exists"):
        sys.exit(f"[error] Unknown or missing model alias '{alias}'")
        
    new_config = {}
    if args.n_ctx:
        new_config["configured_context_length"] = args.n_ctx
    if args.max_input:
        new_config["max_input_tokens"] = "auto" if args.max_input == "auto" else int(args.max_input)
    if args.max_output:
        new_config["max_output_tokens"] = "auto" if args.max_output == "auto" else int(args.max_output)
    if args.temp is not None:
        new_config["temperature"] = args.temp
    if args.top_p is not None:
        new_config["top_p"] = args.top_p
    if args.top_k is not None:
        new_config["top_k"] = args.top_k
        
    if new_config:
        current = get_model_config(alias)
        current.update(new_config)
        
        ctx = current.get("configured_context_length") or caps["total_context_length"]
        inp = current.get("max_input_tokens")
        out = current.get("max_output_tokens")
        
        v_inp = 0 if inp == "auto" else (inp or 0)
        v_out = 0 if out == "auto" else (out or 0)
        
        if v_inp + v_out > ctx:
            sys.exit(f"[error] Invalid Config: max_input ({v_inp}) + max_output ({v_out}) exceeds n_ctx ({ctx})")
            
        set_model_config(alias, current)
        print(f"[config] Updated settings for '{alias}'.")

    config = get_model_config(alias)
    
    disp_ctx = config.get("configured_context_length") or caps["total_context_length"]
    disp_inp = config.get("max_input_tokens")
    disp_out = config.get("max_output_tokens")
    
    if disp_inp == "auto" and disp_out == "auto":
        v_inp = disp_ctx // 2
        v_out = disp_ctx - v_inp
    elif disp_inp == "auto":
        v_out = int(disp_out or 512)
        v_inp = disp_ctx - v_out
    elif disp_out == "auto":
        v_inp = int(disp_inp or 512)
        v_out = disp_ctx - v_inp
    else:
        v_inp, v_out = disp_inp, disp_out

    print(f"\n\033[1mMODEL CAPABILITIES (read-only)\033[0m")
    print(f"{'  Alias':<30} : {caps['alias']}")
    print(f"{'  Path':<30} : {caps['path']}")
    print(f"{'  Size':<30} : {caps['size']}")
    print(f"{'  Total Context Limit':<30} : {caps['total_context_length']} tokens")

    print(f"\n\033[1mUSER CONFIGURATION (editable)\033[0m")
    print(f"{'  Configured Context (n-ctx)':<30} : {disp_ctx} tokens")
    print(f"{'  Input Token Limit':<30} : {disp_inp} {'(' + str(v_inp) + ' resolved)' if disp_inp == 'auto' else ''}")
    print(f"{'  Output Token Limit':<30} : {disp_out} {'(' + str(v_out) + ' resolved)' if disp_out == 'auto' else ''}")
    
    print(f"\n\033[1mSAMPLING PARAMETERS (editable)\033[0m")
    print(f"{'  Temperature':<30} : {config.get('temperature', 0.7)}")
    print(f"{'  Top-P':<30} : {config.get('top_p', 0.9)}")
    print(f"{'  Top-K':<30} : {config.get('top_k', 40)}")

    print(f"\n\033[90mTip: Use --n-ctx, --max-output, --temp, etc., to change these.\033[0m\n")


# ---------------------------------------------------------------------------
# Subcommand: bench
# ---------------------------------------------------------------------------

def cmd_bench(args: argparse.Namespace) -> None:
    from engine import Engine

    engine = Engine(
        alias=args.alias,
        n_gpu_layers=args.gpu_layers,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_threads=args.n_threads,
    )

    prompt = args.prompt
    runs = args.runs
    print(f"[bench] {runs} runs × prompt='{prompt[:60]}'  max_tokens={args.max_tokens}\n")

    tok_secs = []
    for i in range(1, runs + 1):
        t0 = time.perf_counter()
        raw = engine._llm(
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=0.0,
            echo=False,
            stream=False,
        )
        elapsed = time.perf_counter() - t0
        usage = raw.get("usage", {})
        n = usage.get("completion_tokens", args.max_tokens)
        ts = n / elapsed if elapsed else 0
        tok_secs.append(ts)
        print(f"  Run {i:>2}: {n} tokens  {ts:.1f} tok/s  ({elapsed:.2f}s)")

    avg = sum(tok_secs) / len(tok_secs)
    peak = max(tok_secs)
    print(f"\n[bench] Average: {avg:.1f} tok/s   Peak: {peak:.1f} tok/s")
    print(f"[bench] n_batch={args.n_batch}  n_threads={args.n_threads}  "
          f"n_gpu_layers={args.gpu_layers or 'auto'}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _add_perf_args(p: argparse.ArgumentParser) -> None:
    import multiprocessing
    import os
    p.add_argument("--gpu-layers", default=None,
                   help="GPU layers to offload (0=CPU, -1=all, max=auto-detect best fit)")
    p.add_argument("--n-ctx", type=int, default=None,
                   help="Context window size (persisted default available via 'config')")
    p.add_argument("--n-batch", type=int, default=None,
                   help="Token batch size (default: 512)")
    p.add_argument("--n-threads", type=int, default=None,
                   help="CPU threads (default: min(cpus, 8))")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Max tokens to generate (persisted default available via 'config')")
    p.add_argument("--temp", type=float, default=None,
                   help="Sampling temperature (default: 0.7 or persisted value)")
    p.add_argument("--top-p", type=float, default=None,
                   help="Top-P sampling (default: 0.9 or persisted value)")
    p.add_argument("--top-k", type=int, default=None,
                   help="Top-K sampling (default: 40 or persisted value)")
    p.add_argument("--chat", action="store_true",
                   help="Apply the model's internal chat template to the prompt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engine",
        description="High-performance GGUF inference engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_load = sub.add_parser("load", help="Download & register a GGUF model")
    p_load.add_argument("source",
                        help="HF repo id  (e.g. 'TheBloke/Mistral-7B-v0.1-GGUF')  "
                             "or full hf:// URI or HTTPS URL or local path")
    p_load.add_argument("--alias", required=True, help="Short name to refer to this model")
    p_load.add_argument("--file", default=None, help="Specific GGUF filename inside the HF repo")
    p_load.set_defaults(func=cmd_load)

    p_run = sub.add_parser("run", help="Generate text for a single prompt")
    p_run.add_argument("alias", help="Model alias")
    p_run.add_argument("--prompt", required=True, help="Input prompt")
    p_run.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    _add_perf_args(p_run)
    p_run.set_defaults(func=cmd_run)

    p_batch = sub.add_parser("batch", help="Run inference over a file of prompts")
    p_batch.add_argument("alias", help="Model alias")
    p_batch.add_argument("--file", required=True, help="Text file with one prompt per line")
    p_batch.add_argument("--output", default=None, help="Write JSON results to this file")
    p_batch.add_argument("--parallel", type=int, default=0, help="Max prompts per parallel sub-batch (0=auto)")
    _add_perf_args(p_batch)
    p_batch.set_defaults(func=cmd_batch)

    p_chat = sub.add_parser("chat", help="Interactive chat REPL")
    p_chat.add_argument("alias", help="Model alias")
    p_chat.add_argument("--system", default="You are a helpful assistant.", help="System prompt")
    _add_perf_args(p_chat)
    p_chat.set_defaults(func=cmd_chat)

    p_list = sub.add_parser("list", help="Show all registered models")
    p_list.set_defaults(func=cmd_list)

    p_config = sub.add_parser("config", help="View or modify model configuration")
    p_config.add_argument("alias", help="Model alias")
    p_config.add_argument("--n-ctx", type=int, help="Adjust total context window")
    p_config.add_argument("--max-input", help="Set max input tokens (or 'auto')")
    p_config.add_argument("--max-output", help="Set max output tokens (or 'auto')")
    p_config.add_argument("--temp", type=float, help="Set default generation temperature")
    p_config.add_argument("--top-p", type=float, help="Set default Top-P sampling")
    p_config.add_argument("--top-k", type=int, help="Set default Top-K sampling")
    p_config.set_defaults(func=cmd_config)

    p_bench = sub.add_parser("bench", help="Throughput benchmark")
    p_bench.add_argument("alias", help="Model alias")
    p_bench.add_argument("--prompt", default="Tell me a short story about a robot.", help="Prompt to repeat")
    p_bench.add_argument("--runs", type=int, default=3, help="Number of benchmark runs (default: 3)")
    _add_perf_args(p_bench)
    p_bench.set_defaults(func=cmd_bench)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n[engine] Process interrupted.")
    except Exception as e:
        print(f"\n[error] Engine failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
cli.py — Command-line interface for the llama.cpp inference engine.

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
    dest = MODELS_DIR / f"{alias}.gguf"

    # ---- HuggingFace repo  (hf://<org/repo> or plain <org/repo>) ----------
    if source.startswith("hf://") or (
        "/" in source and not source.startswith("http")
    ):
        repo_id = source.removeprefix("hf://")
        _download_hf(repo_id, dest, filename)

    # ---- Direct HTTPS URL -------------------------------------------------
    elif source.startswith("http://") or source.startswith("https://"):
        _download_url(source, dest)

    # ---- Local file path --------------------------------------------------
    elif Path(source).exists():
        import shutil
        print(f"[load] Copying local file → {dest}")
        shutil.copy2(source, dest)

    else:
        sys.exit(f"[error] Cannot resolve source: '{source}'")

    register_model(alias, str(dest), source)
    print(f"[load] Done. Use:  python cli.py run {alias} --prompt \"Hello\"")


def _download_hf(repo_id: str, dest: Path, filename: str | None) -> None:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        sys.exit("[error] huggingface_hub not installed. Run: pip install huggingface_hub")

    # Auto-pick the GGUF file if no explicit filename given
    if not filename:
        all_files = list(list_repo_files(repo_id))
        gguf_files = [f for f in all_files if f.endswith(".gguf")]
        if not gguf_files:
            sys.exit(f"[error] No .gguf files found in {repo_id}. Files:\n" + "\n".join(all_files))
        # Prefer Q4_K_M → Q4_K_S → first available
        preferred = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0"]
        filename = gguf_files[0]  # fallback
        for tag in preferred:
            match = [f for f in gguf_files if tag in f]
            if match:
                filename = match[0]
                break
        print(f"[load] Auto-selected: {filename}  (from {repo_id})")

    print(f"[load] Downloading {repo_id}/{filename} …")
    tmp_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest.parent),
        local_dir_use_symlinks=False,
    )
    # hf_hub_download puts it at <local_dir>/<filename>; rename to <alias>.gguf
    dl = Path(tmp_path)
    if dl != dest:
        dl.rename(dest)
    print(f"[load] Saved → {dest}  ({dest.stat().st_size / 1e9:.2f} GB)")


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
    from engine import Engine

    engine = Engine(
        alias=args.alias,
        n_gpu_layers=args.gpu_layers,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_threads=args.n_threads,
    )

    prompt = args.prompt
    t0 = time.perf_counter()

    # Always stream for a better UX unless explicitly disabled
    stream = not args.no_stream 
    
    print("\n" + "─" * 50)
    tokens = 0
    
    if stream:
        for tok in engine.generate(prompt, max_tokens=args.max_tokens,
                                   temperature=args.temp, stream=True):
            print(tok, end="", flush=True)
            tokens += 1
            
        elapsed = time.perf_counter() - t0
        tps = tokens / elapsed if elapsed > 0 else 0
        print("\n" + "─" * 50)
        print(f"\033[90m[{tokens} tokens | {tps:.1f} tok/s | {elapsed:.2f}s]\033[0m")
    else:
        output = engine.generate(prompt, max_tokens=args.max_tokens,
                                 temperature=args.temp, stream=False)
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

    engine = Engine(
        alias=args.alias,
        n_gpu_layers=args.gpu_layers,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_threads=args.n_threads,
    )

    results = engine.generate_batch(
        prompts,
        max_tokens=args.max_tokens,
        temperature=args.temp,
    )

    # Optionally write results to JSON
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
    from engine import Engine

    engine = Engine(
        alias=args.alias,
        n_gpu_layers=args.gpu_layers,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_threads=args.n_threads,
    )
    engine.chat(
        system_prompt=args.system,
        max_tokens=args.max_tokens,
        temperature=args.temp,
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
        size = f"{p.stat().st_size / 1e9:.2f}G" if p.exists() else "MISSING"
        print(f"{alias:<20}  {size:>8}  {meta['source']}")
    print()


# ---------------------------------------------------------------------------
# Subcommand: info
# ---------------------------------------------------------------------------

def cmd_info(args: argparse.Namespace) -> None:
    from engine import list_models

    reg = list_models()
    alias = args.alias
    if alias not in reg:
        sys.exit(f"[error] Unknown alias '{alias}'")

    meta = reg[alias]
    p = Path(meta["path"])
    print(f"\nAlias  : {alias}")
    print(f"Source : {meta['source']}")
    print(f"Path   : {meta['path']}")
    print(f"Exists : {p.exists()}")
    if p.exists():
        print(f"Size   : {p.stat().st_size / 1e9:.3f} GB")
    print()


# ---------------------------------------------------------------------------
# Subcommand: bench
# ---------------------------------------------------------------------------

def cmd_bench(args: argparse.Namespace) -> None:
    """
    Quick throughput benchmark — runs the same prompt N times and reports
    average tokens/sec.  Useful for tuning n_batch / n_gpu_layers.
    """
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
            temperature=0.0,   # greedy for reproducibility
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
    """Common performance-tuning flags shared by run/batch/chat/bench."""
    import multiprocessing, os
    p.add_argument("--gpu-layers", type=int, default=None,
                   help="GPU layers to offload (0=CPU, None=auto-detect)")
    p.add_argument("--n-ctx", type=int,
                   default=int(os.environ.get("LLM_N_CTX", "4096")),
                   help="Context window size (default: 4096)")
    p.add_argument("--n-batch", type=int,
                   default=int(os.environ.get("LLM_N_BATCH", "512")),
                   help="Token batch size — larger = more throughput (default: 512)")
    p.add_argument("--n-threads", type=int,
                   default=int(os.environ.get("LLM_N_THREADS",
                               str(min(multiprocessing.cpu_count(), 8)))),
                   help="CPU threads (default: min(cpus, 8))")
    p.add_argument("--max-tokens", type=int, default=512,
                   help="Max tokens to generate (default: 512)")
    p.add_argument("--temp", type=float, default=0.7,
                   help="Sampling temperature (default: 0.7)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engine",
        description="High-performance llama.cpp inference engine (GGUF)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- load ----
    p_load = sub.add_parser("load", help="Download & register a GGUF model")
    p_load.add_argument("source",
                        help="HF repo id  (e.g. 'TheBloke/Mistral-7B-v0.1-GGUF')  "
                             "or full hf:// URI or HTTPS URL or local path")
    p_load.add_argument("--alias", required=True, help="Short name to refer to this model")
    p_load.add_argument("--file", default=None,
                        help="Specific GGUF filename inside the HF repo")
    p_load.set_defaults(func=cmd_load)

    p_run = sub.add_parser("run", help="Generate text for a single prompt")
    p_run.add_argument("alias", help="Model alias")
    p_run.add_argument("--prompt", required=True, help="Input prompt")
    p_run.add_argument("--no-stream", action="store_true",
                       help="Disable streaming output")
    _add_perf_args(p_run)
    p_run.set_defaults(func=cmd_run)

    # ---- batch ----
    p_batch = sub.add_parser("batch", help="Run inference over a file of prompts")
    p_batch.add_argument("alias", help="Model alias")
    p_batch.add_argument("--file", required=True,
                         help="Text file with one prompt per line")
    p_batch.add_argument("--output", default=None,
                         help="Write JSON results to this file")
    _add_perf_args(p_batch)
    p_batch.set_defaults(func=cmd_batch)

    # ---- chat ----
    p_chat = sub.add_parser("chat", help="Interactive chat REPL")
    p_chat.add_argument("alias", help="Model alias")
    p_chat.add_argument("--system", default="You are a helpful assistant.",
                        help="System prompt")
    _add_perf_args(p_chat)
    p_chat.set_defaults(func=cmd_chat)

    # ---- list ----
    p_list = sub.add_parser("list", help="Show all registered models")
    p_list.set_defaults(func=cmd_list)

    # ---- info ----
    p_info = sub.add_parser("info", help="Show details for one alias")
    p_info.add_argument("alias", help="Model alias")
    p_info.set_defaults(func=cmd_info)

    # ---- bench ----
    p_bench = sub.add_parser("bench", help="Throughput benchmark")
    p_bench.add_argument("alias", help="Model alias")
    p_bench.add_argument("--prompt", default="Tell me a short story about a robot.",
                         help="Prompt to repeat")
    p_bench.add_argument("--runs", type=int, default=3,
                          help="Number of benchmark runs (default: 3)")
    _add_perf_args(p_bench)
    p_bench.set_defaults(func=cmd_bench)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

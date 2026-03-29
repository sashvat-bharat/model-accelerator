"""
cli.py — Command-line interface for the GGUF inference engine.

Usage:
  python cli.py load <hf_repo_or_url> --alias <name> [--file <filename>]
  python cli.py run  <alias> --prompt "text" [--max-tokens N] [--temp F] [--stream]
  python cli.py batch <alias> --file prompts.txt [--max-tokens N]
  python cli.py chat  <alias> [--system "You are..."] [--max-tokens N]
  python cli.py list
  python cli.py config <alias>
  python cli.py bench <alias> [--prompt "text"] [--runs N]

Environment variables for performance tuning:
  GGUF_MODELS_DIR — Custom directory to store models
  LLM_N_CTX      — Context window size          (default: 4096)
  LLM_N_BATCH    — Token batch size             (default: 512)
  LLM_N_THREADS  — CPU thread count             (default: min(cpus, 8))
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Subcommand: load
# ---------------------------------------------------------------------------

def cmd_load(args: argparse.Namespace) -> None:
    from engine import MODELS_DIR, register_model

    source: str = args.source
    alias: str = args.alias
    filename: str = args.file

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest_dir = MODELS_DIR / alias

    if source.startswith("hf://") or ("/" in source and not source.startswith("http")):
        repo_id = source.removeprefix("hf://")
        _download_hf(repo_id, dest_dir, filename)
        register_model(alias, str(dest_dir), source)

    elif source.startswith("http://") or source.startswith("https://"):
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"{alias}.gguf"
        _download_url(source, dest_file)
        register_model(alias, str(dest_dir), source)

    elif Path(source).exists():
        import shutil
        dest_dir.mkdir(parents=True, exist_ok=True)
        if Path(source).is_dir():
            console.print(f"[dim]Copying directory → {dest_dir}[/]")
            shutil.copytree(source, dest_dir, dirs_exist_ok=True)
        else:
            console.print(f"[dim]Copying file → {dest_dir / Path(source).name}[/]")
            shutil.copy2(source, dest_dir / Path(source).name)
        register_model(alias, str(dest_dir), source)

    else:
        console.print(f"[bold red][error][/] Cannot resolve source: '{source}'")
        sys.exit(1)

    console.print(f"[bold green]Done.[/] Use: [cyan]python cli.py run {alias} --prompt \"Hello\" --chat[/]")


def _download_hf(repo_id: str, dest_dir: Path, filename: str | None) -> Path:
    try:
        from huggingface_hub import snapshot_download, list_repo_files
    except ImportError:
        console.print("[bold red][error][/] huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    all_files = list(list_repo_files(repo_id))
    gguf_files = [f for f in all_files if f.endswith(".gguf")]
    if not gguf_files:
        console.print(f"[bold red][error][/] No .gguf files found in {repo_id}")
        sys.exit(1)

    if not filename:
        preferred = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0"]
        filename = gguf_files[0]
        for tag in preferred:
            match = [f for f in gguf_files if tag in f]
            if match:
                filename = match[0]
                break
        console.print(f"[dim]Auto-selected GGUF: {filename}[/]")

    other_ggufs = [f for f in gguf_files if f != filename]
    console.print(f"[dim]Downloading {repo_id} (GGUF: {filename}) to {dest_dir} …[/]")
    
    final_path = snapshot_download(
        repo_id=repo_id, local_dir=str(dest_dir), allow_patterns=[filename],
        ignore_patterns=other_ggufs, local_dir_use_symlinks=False,
    )
    return Path(final_path) / filename

def _download_url(url: str, dest: Path) -> None:
    import urllib.request
    console.print(f"[dim]Downloading {url} …[/]")

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(block_num * block_size / total_size * 100, 100)
            print(f"\r  {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    console.print(f"\n[bold green]Saved → {dest}[/]")


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    from engine import Engine, get_model_config

    config_data = get_model_config(args.alias)
    n_ctx = args.n_ctx or config_data.configured_context_length
    
    engine = Engine(
        alias=args.alias, n_gpu_layers=args.gpu_layers,
        n_ctx=n_ctx, n_batch=args.n_batch, n_threads=args.n_threads,
    )

    prompt = args.prompt
    stream = not args.no_stream 
    disp_max = args.max_tokens or config_data.max_output_tokens
    disp_temp = args.temp if args.temp is not None else config_data.temperature
    disp_top_p = args.top_p if args.top_p is not None else config_data.top_p
    disp_top_k = args.top_k if args.top_k is not None else config_data.top_k
    
    console.rule("[bold cyan]GENERATION[/]")
    console.print(f"  [bold]Prompt[/]     : \"{prompt[:60]}{'...' if len(prompt)>60 else ''}\"")
    console.print(f"  [bold]Temp[/]       : {disp_temp:.2f}")
    console.print(f"  [bold]Max Tokens[/] : {disp_max}")
    console.print(f"  [bold]Mode[/]       : {'Streaming' if stream else 'Static'} ({'Chat' if args.chat else 'Completion'})")
    console.rule()

    t0 = time.perf_counter()
    tokens = 0

    if stream:
        for tok in engine.generate(prompt, max_tokens=args.max_tokens,
                                   temperature=disp_temp, top_p=disp_top_p, top_k=disp_top_k,
                                   stream=True, chat_format=args.chat):
            # We use standard print here so rich doesn't crash trying to parse LLM-generated brackets
            print(tok, end="", flush=True)
            tokens += 1
            
        elapsed = time.perf_counter() - t0
        tps = tokens / elapsed if elapsed > 0 else 0
        console.rule()
        console.print(f"[dim][{tokens} tokens | {tps:.1f} tok/s | {elapsed:.2f}s][/]")
    else:
        output = engine.generate(prompt, max_tokens=args.max_tokens,
                                 temperature=disp_temp, top_p=disp_top_p, top_k=disp_top_k,
                                 stream=False, chat_format=args.chat)
        elapsed = time.perf_counter() - t0
        console.print(output)
        console.rule()
        console.print(f"[dim][{elapsed:.2f}s][/]")


# ---------------------------------------------------------------------------
# Subcommand: batch
# ---------------------------------------------------------------------------

def cmd_batch(args: argparse.Namespace) -> None:
    from engine import Engine, get_model_config

    pfile = Path(args.file)
    if not pfile.exists():
        console.print(f"[bold red][error][/] Prompt file not found: {args.file}")
        sys.exit(1)

    prompts = [line.strip() for line in pfile.read_text().splitlines() if line.strip()]
    if not prompts:
        console.print("[bold red][error][/] No prompts found in file.")
        sys.exit(1)

    config_data = get_model_config(args.alias)
    n_ctx = args.n_ctx or config_data.configured_context_length

    console.rule("[bold cyan]BATCH GENERATION[/]")
    console.print(f"  [bold]Source[/]   : {args.file} ({len(prompts)} prompts)")
    console.print(f"  [bold]Parallel[/] : {args.parallel if args.parallel > 0 else 'auto'}")
    console.rule()

    engine = Engine(
        alias=args.alias, n_gpu_layers=args.gpu_layers,
        n_ctx=n_ctx, n_batch=args.n_batch, n_threads=args.n_threads,
    )

    results = engine.generate_batch(
        prompts, max_tokens=args.max_tokens, temperature=args.temp,
        top_p=args.top_p, top_k=args.top_k, parallel=args.parallel, chat_format=args.chat,
    )

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        console.print(f"\n[bold green]Results written to {args.output}[/]")
    else:
        console.rule("[bold cyan]Results[/]")
        for r in results:
            console.print(f"\n[bold]Prompt[/] : {r['prompt'][:80]}…")
            console.print(f"[bold]Output[/] : {r['output'][:200]}…")
            console.print(f"[dim]Stats  : {r['tokens']} tokens, {r['tok_per_sec']} tok/s[/]")


# ---------------------------------------------------------------------------
# Subcommand: chat
# ---------------------------------------------------------------------------

def cmd_chat(args: argparse.Namespace) -> None:
    from engine import Engine, get_model_config

    config_data = get_model_config(args.alias)
    engine = Engine(
        alias=args.alias, n_gpu_layers=args.gpu_layers,
        n_ctx=args.n_ctx or config_data.configured_context_length,
        n_batch=args.n_batch, n_threads=args.n_threads,
    )
    
    engine.chat(
        system_prompt=args.system, max_tokens=args.max_tokens,
        temperature=args.temp, top_p=args.top_p, top_k=args.top_k,
    )


# ---------------------------------------------------------------------------
# Subcommand: list
# ---------------------------------------------------------------------------
def cmd_list(_args: argparse.Namespace) -> None:
    from engine import list_models

    reg = list_models()
    if not reg:
        console.print("[yellow]No models registered yet. Run: python cli.py load <repo> --alias <name>[/]")
        return

    table = Table(title="Registered Models", header_style="bold cyan")
    table.add_column("Alias", style="bold")
    table.add_column("Size", justify="right")
    table.add_column("Source", style="dim")

    for alias, meta in reg.items():
        p = Path(meta["path"])
        size = "MISSING"
        if p.exists():
            if p.is_dir():
                size = f"{sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / 1e9:.2f}G"
            else:
                size = f"{p.stat().st_size / 1e9:.2f}G"
        table.add_row(alias, size, meta["source"])

    console.print(table)


# ---------------------------------------------------------------------------
# Subcommand: config
# ---------------------------------------------------------------------------

def cmd_config(args: argparse.Namespace) -> None:
    from engine import get_model_capabilities, get_model_config, set_model_config
    
    alias = args.alias
    caps = get_model_capabilities(alias)
    if not caps.exists:
        console.print(f"[bold red][error][/] Unknown or missing model alias '{alias}'")
        sys.exit(1)
        
    current = get_model_config(alias)
    updated = False

    if args.n_ctx is not None: current.configured_context_length = args.n_ctx; updated = True
    if args.max_input: current.max_input_tokens = "auto" if args.max_input == "auto" else int(args.max_input); updated = True
    if args.max_output: current.max_output_tokens = "auto" if args.max_output == "auto" else int(args.max_output); updated = True
    if args.temp is not None: current.temperature = args.temp; updated = True
    if args.top_p is not None: current.top_p = args.top_p; updated = True
    if args.top_k is not None: current.top_k = args.top_k; updated = True
        
    if updated:
        set_model_config(alias, current)
        console.print(f"[bold green][config] Updated settings for '{alias}'.[/]\n")

    disp_ctx = current.configured_context_length or caps.total_context_length
    
    console.print("[bold cyan]MODEL CAPABILITIES (read-only)[/]")
    console.print(f"  [bold]Alias[/]                 : {caps.alias}")
    console.print(f"  [bold]Total Layers[/]          : {caps.total_layers} (Excl. LM Head)")
    console.print(f"  [bold]Max Native Context[/]    : {caps.total_context_length} tokens")

    console.print("\n[bold cyan]USER CONFIGURATION (editable)[/]")
    console.print(f"  [bold]n-ctx (Configured)[/]    : {disp_ctx} tokens")
    console.print(f"  [bold]Max Input Tokens[/]      : {current.max_input_tokens}")
    console.print(f"  [bold]Max Output Tokens[/]     : {current.max_output_tokens}")
    
    console.print("\n[bold cyan]SAMPLING PARAMETERS (editable)[/]")
    console.print(f"  [bold]Temperature[/]           : {current.temperature}")
    console.print(f"  [bold]Top-P[/]                 : {current.top_p}")
    console.print(f"  [bold]Top-K[/]                 : {current.top_k}")

    console.print("\n[dim]Tip: Use --n-ctx, --max-output, --temp, etc., to change these.[/]\n")


# ---------------------------------------------------------------------------
# Subcommand: bench
# ---------------------------------------------------------------------------

def cmd_bench(args: argparse.Namespace) -> None:
    from engine import Engine

    engine = Engine(
        alias=args.alias, n_gpu_layers=args.gpu_layers,
        n_ctx=args.n_ctx, n_batch=args.n_batch, n_threads=args.n_threads,
    )

    console.rule("[bold cyan]BENCHMARK[/]")
    console.print(f"Runs: {args.runs} | Prompt: '{args.prompt[:40]}...'\n")

    tok_secs = []
    for i in range(1, args.runs + 1):
        t0 = time.perf_counter()
        raw = engine._llm(prompt=args.prompt, max_tokens=args.max_tokens, temperature=0.0, echo=False)
        elapsed = time.perf_counter() - t0
        
        n = raw.get("usage", {}).get("completion_tokens", args.max_tokens)
        ts = n / elapsed if elapsed else 0
        tok_secs.append(ts)
        console.print(f"  Run {i:>2}: {n} tokens  [bold green]{ts:.1f} tok/s[/]  ({elapsed:.2f}s)")

    avg = sum(tok_secs) / len(tok_secs)
    peak = max(tok_secs)
    console.rule()
    console.print(f"[bold]Average:[/] [cyan]{avg:.1f} tok/s[/]   [bold]Peak:[/] [cyan]{peak:.1f} tok/s[/]")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _add_perf_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gpu-layers", default=None,
                   help="GPU layers to offload ('max' = full auto-offload, integer = exact layers)")
    p.add_argument("--n-ctx", type=int, default=None, help="Context window size")
    p.add_argument("--n-batch", type=int, default=None, help="Token batch size (default: 512)")
    p.add_argument("--n-threads", type=int, default=None, help="CPU threads")
    p.add_argument("--max-tokens", type=int, default=None, help="Max tokens to generate")
    p.add_argument("--temp", type=float, default=None, help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=None, help="Top-P sampling")
    p.add_argument("--top-k", type=int, default=None, help="Top-K sampling")
    p.add_argument("--chat", action="store_true", help="Apply model's native chat template")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engine", description="High-performance GGUF inference engine",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_load = sub.add_parser("load", help="Download & register a GGUF model")
    p_load.add_argument("source", help="HF repo id, full URL, or local path")
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

    p_list = sub.add_parser("list", help="Show all registered models").set_defaults(func=cmd_list)

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
        console.print("\n[dim]Process interrupted.[/]")
    except Exception as e:
        console.print(f"\n[bold red][error][/] Engine failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
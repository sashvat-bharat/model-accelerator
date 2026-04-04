# Fix: Color thinking tags and add line breaks in engine.py chat() method

## Context
The --think flag is already implemented and working. The model generates thinking content.
The issue is in engine.py chat() method (lines ~891-933) -- the thinking tags need:
1. Colored output (bold magenta for tags, dim cyan for content)
2. Line breaks after the opening tag and before the closing tag

## What is already correct
- cli.py streaming mode (cmd_run, lines ~211-253): Already fixed with proper colors and line breaks
- cli.py non-streaming mode (_format_thinking_output, lines ~34-67): Already fixed

## What needs fixing
engine.py -- the chat() method thinking-enabled block (approximately lines 906-931).

### Current code (WRONG -- no colors on tags, no line breaks):
The block starting at line 906 handles thinking tag detection but uses plain print() for tags without ANSI color codes and without line breaks.

Specifically these lines need fixing:
- Line 911: print("<think_open>", end="", flush=True) -- needs color and newline
- Line 913-915: console.print(f"[dim cyan]{parts[1]}[/]"...) -- needs newline after tag
- Line 920-922: console.print(f"[dim cyan]{parts[0]}[/]"...) -- needs newline before close tag
- Line 923: print("<think_close>", end="", flush=True) -- needs color and newline
- Line 929: print("<think_open>", end="", flush=True) -- needs color and newline
- Line 930: console.print(f"[dim cyan]{tok}[/]"...) -- needs newline after tag

### The fix
Replace the entire think_enabled block (lines 891-933) with this corrected version:

```python
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
                        if "<think_open>" in tok:
                            thinking_started = True
                            parts = tok.split("<think_open>", 1)
                            if parts[0]:
                                print(parts[0], end="", flush=True)
                            console.print(f"[bold magenta]<think_open>[/]")
                            if parts[1]:
                                console.print(f"[dim cyan]{parts[1]}[/]", end="", flush=True)
                            continue
                        elif "<think_close>" in tok:
                            parts = tok.split("<think_close>", 1)
                            if parts[0]:
                                console.print(f"[dim cyan]{parts[0]}[/]")
                            console.print(f"[bold magenta]<think_close>[/]")
                            if parts[1]:
                                print(parts[1], end="", flush=True)
                            continue
                        elif not thinking_started:
                            thinking_started = True
                            console.print(f"[bold magenta]<think_open>[/]")
                            console.print(f"[dim cyan]{tok}[/]", end="", flush=True)
                            continue
                        console.print(f"[dim cyan]{tok}[/]", end="", flush=True)
                        tokens_out.append(tok)
```

### Key changes
1. All print("<think_open>") replaced with console.print(f"[bold magenta]<think_open>[/]") -- adds magenta color and implicit newline
2. All print("<think_close>") replaced with console.print(f"[bold magenta]<think_close>[/]") -- adds magenta color and implicit newline
3. Content before <think_close> uses console.print() without end="" so it gets a newline before the closing tag
4. The fallback case (not thinking_started) also uses console.print for the opening tag with color


#### 1. Load Model

```bash
python cli.py load unsloth/Qwen3.5-9B-GGUF --alias qwen3_5-9b
```

#### 2. Run Model

```bash
python cli.py run qwen3_5-9b --prompt "Explain Entropy." --chat --gpu-layers max
```

#### 3. Chat with Model

```bash
python cli.py chat qwen3_5-9b
```

#### 4. Batch Processing

```bash
python cli.py batch qwen3_5-9b --file prompts.txt --output results.json --parallel 25 --chat --gpu-layers max
```

#### 5. Changing the Config

```bash
python cli.py config qwen3_5-9b --n-ctx 4096 --max-output 512 --max-input auto --temp 0.7 --top-p 1.0 --top-k 0
```
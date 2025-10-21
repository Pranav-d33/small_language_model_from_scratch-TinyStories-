# Small Language Model (SLM) on TinyStories

Train a GPT-style Small Language Model (SLM) from scratch on the TinyStories dataset using PyTorch. The notebook covers dataset loading, tokenization to on-disk memmaps, a compact GPT architecture, training with mixed precision and warmup+cosine scheduling, and text generation.

- Main notebook: `small_language_model_on_tinystories_dataset.ipynb`
- Environment target: Google Colab (GPU runtime)
- Minimum GPU: T4 (or better: P100/V100/A100)

## Highlights

- Dataset: Hugging Face TinyStories (`roneneldan/TinyStories`)
- Tokenization: `tiktoken` GPT-2 encoder; writes `train.bin` and `validation.bin` as NumPy memmaps
- Model: GPT-like Transformer (tied embeddings, GELU MLP, dropout, optional Flash Attention via `scaled_dot_product_attention`)
- Training: AdamW, linear warmup + cosine decay, gradient accumulation, AMP mixed precision, gradient clipping
- Checkpoint: best weights saved to `best_model_params.pt`
- Inference: temperature and optional top-k sampling via `model.generate`

## Run on Google Colab (recommended)

1) Open the notebook in Google Colab.
2) Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU (T4 or better recommended).
3) Verify the GPU is available:
   ```python
   import torch
   print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
   ```
4) Run all cells top-to-bottom. Installation lines must use the `!` prefix in Colab.
   - In this notebook, change `pip install -U datasets` to `!pip install -U datasets` if needed.
5) Keep several GB of free disk space for `train.bin`/`validation.bin` memmaps.

## What the notebook does

1) Import TinyStories with `datasets.load_dataset("roneneldan/TinyStories")`.
2) Tokenize all splits using `tiktoken` and stream IDs to disk as memmaps (`train.bin`, `validation.bin`).
3) Create efficient batches from memmaps with a sliding window of length `block_size`.
4) Define a compact GPT model with config:
   - Example config in-notebook: `block_size=128, n_layer=6, n_head=6, n_embd=384, dropout=0.1, vocab_size=50257`.
5) Configure training: learning rate schedule (warmup → cosine), AMP autocast, GradScaler, gradient clipping.
6) Train and periodically evaluate; save the best checkpoint.
7) Plot train/validation loss.
8) Reload the best checkpoint and generate text from prompts.

Artifacts written next to the notebook:
- `train.bin`, `validation.bin` — token IDs (uint16) as NumPy memmaps
- `best_model_params.pt` — best-performing checkpoint

## Local setup (Windows PowerShell)

Requirements:
- Python 3.10+
- Recommended: NVIDIA GPU with recent CUDA drivers (PyTorch CUDA build)

Create a virtual environment and install dependencies:

```powershell
# from the project folder
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

Notes:
- For CUDA-specific PyTorch wheels, see https://pytorch.org/get-started/locally/ and install the matching `torch` build.

## Run locally in Jupyter (optional)

Minimal adjustments if running outside Colab:
- Ensure notebook install lines use `!pip` inside notebook cells (e.g., `!pip install -U datasets`).
- Remove or comment the Colab-only line at the end: `from google.colab import runtime; runtime.unassign()`.
- Use a CUDA-enabled PyTorch build if you have an NVIDIA GPU.

Then run locally:
1) Open `small_language_model_on_tinystories_dataset.ipynb` in VS Code (Jupyter) or Jupyter Lab/Notebook.
2) Select the Python interpreter for the `.venv` you created.
3) Run cells top-to-bottom:
   - Step 1: Download TinyStories.
   - Step 2: Tokenize and write `train.bin`/`validation.bin` to disk.
   - Step 3: Define `get_batch` to read from memmaps.
   - Step 4–7: Define model + training configuration (optimizer, schedulers, AMP).
   - Step 8: Train; best checkpoint is saved to `best_model_params.pt`.
   - Step 9: Plot losses.
   - Step 10: Inference: load best checkpoint and generate text.

Tip:
- Training parameters in the notebook include `batch_size=32`, `block_size=128`, `gradient_accumulation_steps=32`, `max_iters=20000`, `eval_iters=500`.

## Inference

After training or if `best_model_params.pt` already exists:
- The notebook re-creates the `GPT` with the same `GPTConfig`, loads the checkpoint, and runs `model.generate` on tokenized prompts (e.g., "Once upon a time there was a pumpkin.").
- Sampling uses temperature scaling and optional top-k filtering for diversity.

## Troubleshooting

- GPU not used: In Colab, ensure GPU runtime is selected; locally, verify the selected interpreter and that `torch.cuda.is_available()` returns True. Install a CUDA-enabled PyTorch if you have an NVIDIA GPU.
- Disk space: Memmaps stream data to disk; keep several GB free for `train.bin`/`validation.bin`.
- Tokenizer install: If `tiktoken` fails to build, try upgrading `pip`/`setuptools`/`wheel` first, then reinstall.
- Colab cell: The last cell references `google.colab.runtime.unassign()`; ignore or remove if not running on Colab.

## Acknowledgements

- Dataset: TinyStories — https://huggingface.co/datasets/roneneldan/TinyStories
- Utility patterns inspired by Karpathy's nanoGPT (tokenization and training structure)

## License

No explicit license provided. Add a `LICENSE` file if you plan to distribute or share the model or code.

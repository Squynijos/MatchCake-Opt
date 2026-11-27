# Plan: nanochat-style LLM pipeline for fermionic (MatchCake) models

Status: **approved** (review answers recorded in section 5).

## 1. Goal

Bring the training and evaluation logic of [karpathy/nanochat](https://github.com/karpathy/nanochat) into
MatchCake-Opt so that a **fermionic language model** (built with MatchCake) can be pretrained, fine-tuned for
chat and evaluated with the same workflow as nanochat, on **one GPU node** (1 to 8 GPUs).

Every component of a run is **swappable from a run config file**: tokenizer, model, datasets, optimizer,
LR schedule, evaluations, logger, hardware.

### Decisions already taken

| Topic | Decision |
|---|---|
| Model | Provided by the user. Contract: takes token ids `(B, T)` and returns what the pipeline needs (logits `(B, T, V)`); a small mapper can adapt it. MatchCake-Opt stays quantum-only: no classical GPT is shipped. Classical baselines are run with the original nanochat repo. |
| Stages, in order | 1. pretraining + val bits-per-byte (bpb) → 2. chat SFT + chat evaluation → 3. RL (later). |
| Metrics | bpb, loss, per-task accuracy. **No CORE / ChatCORE aggregates.** |
| Tokenizer | Plug in an **already trained** tokenizer (no tokenizer training stage). |
| Framework | PyTorch Lightning (already used by MatchCake-Opt). |
| Scale | Same target as nanochat: one GPU node, multi-GPU via DDP. |
| Workflow | One feature per branch, PRs into `dev` on `Squynijos/MatchCake-Opt`. |

## 2. What nanochat does (reference)

| nanochat file | Role | Kept? |
|---|---|---|
| `scripts/base_train.py` | step-based pretraining loop: grad accumulation to a token budget, warmup/constant/warmdown LR, eval every N steps, sampling, checkpoints | **yes**, mapped onto Lightning |
| `nanochat/dataloader.py`, `dataset.py` | parquet shards (ClimbMix), BOS-aligned best-fit packing, DDP sharding, approximate resume | **yes** |
| `nanochat/loss_eval.py` | bits per byte (tokenizer-independent loss) | **yes** |
| `nanochat/optim.py` | `MuonAdamW` (Muon on 2D matrices, AdamW elsewhere, ZeRO-2 sharding) | **no**: AdamW (any `torch.optim` class from the config) |
| `nanochat/tokenizer.py` | tiktoken/rustbpe tokenizer, chat template rendering with loss mask, `token_bytes` | interface + adapters, **no training** |
| `scripts/chat_sft.py`, `tasks/*` | SFT on a task mixture (SmolTalk, MMLU aux, GSM8K) with masked targets | **yes** |
| `scripts/chat_eval.py`, `nanochat/engine.py`, `execution.py` | categorical eval (ARC, MMLU) via answer-letter logits; generative eval (GSM8K, HumanEval) via sampling; sandboxed code execution | **yes**, generation without KV cache by default |
| `scripts/chat_rl.py` | GRPO-style RL on GSM8K | **later** (phase 4) |
| `nanochat/core_eval.py`, `scripts/base_eval.py` (CORE) | CORE benchmark | **no** |
| `scripts/tok_train.py`, `tok_eval.py` | tokenizer training | **no** |
| `nanochat/gpt.py`, `flash_attention.py`, `fp8.py` | classical GPT and its speed tricks | **no** (quantum-only) |
| depth-based auto-scaling of width/LR/horizon | GPT-specific | replaced by explicit config values; optional `target_param_data_ratio` to derive `max_steps` |

## 3. Architecture

```
run config (YAML)
   │
   ▼
matchcake_opt CLI  (LightningCLI: fit / validate / test / chat-eval)
   │
   ├── tokenizer      : Tokenizer            (tiktoken | HuggingFace | nanochat dir)
   ├── data           : LightningDataModule  (PretrainDataModule | SFTDataModule)
   │       └── sources: TextSource / Task    (parquet shards | HF dataset | local files | SmolTalk, MMLU, ...)
   ├── model          : LanguageModel(BaseModel)
   │       └── backbone: any torch.nn.Module  (the user's fermionic model, ids -> logits)
   ├── optimizer      : AdamW (default) | any torch.optim class
   ├── lr_scheduler   : warmup-constant-warmdown | cosine | any
   ├── callbacks      : sampling, chat eval, checkpointing
   ├── logger         : TensorBoard (default) | Weights & Biases | CSV
   └── trainer        : devices, strategy (ddp), precision, max_steps, val every N steps, ...
```

### 3.1 Model contract

The user's model is a plain `torch.nn.Module` passed as `backbone`:

```python
class MyFermionicLM(torch.nn.Module):
    def forward(self, idx: torch.LongTensor) -> torch.Tensor:  # (B, T) -> (B, T, vocab_size)
        ...
```

Optional hooks the pipeline uses **if present** (never required):

| Hook | Used for |
|---|---|
| `param_groups(**lrs)` | custom optimizer parameter groups (e.g. different LR or weight decay per group) |
| `generate_step(idx, cache)` | faster incremental generation; otherwise generation recomputes the full sequence each token |
| `num_flops_per_token()` | MFU / throughput logging |

`LanguageModel` (a `BaseModel` subclass, so it stays compatible with the existing pipelines) owns everything
that is not the model: cross-entropy with ignore index `-1`, bpb metric, logging, optimizer/scheduler wiring,
loading weights from a previous stage checkpoint (`init_from: path.ckpt`), and generation (temperature, top-k).

### 3.2 Tokenizer interface

```python
class Tokenizer(Protocol):
    vocab_size: int
    bos_token_id: int
    def encode(self, text: str | list[str]) -> list[int] | list[list[int]]: ...
    def decode(self, ids: list[int]) -> str: ...
    def encode_special(self, name: str) -> int: ...
    def token_bytes(self) -> torch.Tensor: ...           # bytes per token id, 0 for specials (for bpb)
    def render_conversation(self, conv, max_tokens) -> tuple[list[int], list[int]]: ...  # ids, loss mask
    def render_for_completion(self, conv) -> list[int]: ...
```

Adapters:
- `TiktokenTokenizer` — named encodings (`gpt2`, `cl100k_base`, `o200k_base`, ...).
- `HFTokenizer` — `tokenizers` / `transformers` from a Hub id or a local `tokenizer.json`.
- `NanochatTokenizer` — a tokenizer directory produced by nanochat's `tok_train.py`.

The chat special tokens (`<|user_start|>`, `<|assistant_start|>`, `<|python_start|>`, ...) are added on top of a
pretrained vocabulary when they are missing, so `vocab_size` grows accordingly and is passed to the model
automatically (config linking, see 3.5).

### 3.3 Data

**Pretraining** — `PretrainDataModule(tokenizer, train_source, val_source, seq_len, device_batch_size, ...)`:
- `TextSource` implementations: `ParquetShards` (nanochat's ClimbMix shards, with a download helper),
  `HFTextDataset` (`datasets`, streaming), `TextFiles` (local `.txt`/`.jsonl`).
- `PackedTokenStream` (an `IterableDataset`): tokenizes on the fly and does nanochat's BOS-aligned best-fit
  packing; shards by DDP rank **and** dataloader worker; exposes `state_dict()` so a resumed run continues
  roughly where it stopped (as nanochat does).

**SFT** — `SFTDataModule(tokenizer, train_tasks, val_tasks, ...)`:
- `Task` / `TaskMixture` / `TaskSequence` ported from nanochat; tasks: `SmolTalk`, `MMLU`, `GSM8K`, `ARC`,
  `HumanEval`, plus `CustomJSONConversations` for local data.
- Conversations rendered with the tokenizer's chat template; non-assistant tokens get target `-1`.
- Best-fit packing of conversations, as in `chat_sft.py`.

### 3.4 Training loop mapping (nanochat → Lightning)

| nanochat | Lightning equivalent |
|---|---|
| `--num-iterations` / `--target-param-data-ratio` | `trainer.max_steps` (a helper can derive it from a token budget) |
| `--total-batch-size` (tokens) with grad accumulation | `data.total_batch_size_tokens` → `trainer.accumulate_grad_batches` computed at setup |
| `--eval-every`, `--eval-tokens` | `trainer.val_check_interval`, `trainer.limit_val_batches` |
| `--sample-every` | `SamplingCallback(every_n_steps, prompts)` → text logged to TensorBoard |
| `--save-every`, resume | `ModelCheckpoint(every_n_train_steps)` + `ckpt_path=last` (model, optimizer, scheduler, data state) |
| LR warmup/constant/warmdown | `WarmupWarmdownLR` (`LambdaLR`, `interval: step`) |
| `torchrun` + DDP | `trainer.devices: 8`, `trainer.strategy: ddp` |
| bf16 autocast, `torch.compile` | `trainer.precision`, `model.compile: true` — both optional (see open question Q3) |
| wandb `--run` | `trainer.logger` (TensorBoard by default, wandb optional) |

### 3.5 Run config file

LightningCLI configs: each component is a `class_path` + `init_args`, so swapping a component means editing one
block (or overriding on the command line). The full resolved config is saved next to every run for
reproducibility.

```yaml
# runs/pretrain_example.yaml
seed_everything: 42
tokenizer:
  class_path: matchcake_opt.tokenizers.TiktokenTokenizer
  init_args: {name: gpt2}
data:
  class_path: matchcake_opt.datamodules.PretrainDataModule
  init_args:
    train_source: {class_path: matchcake_opt.datasets.text.ParquetShards, init_args: {split: train, num_shards: 8}}
    val_source:   {class_path: matchcake_opt.datasets.text.ParquetShards, init_args: {split: val}}
    seq_len: 256
    device_batch_size: 32
    total_batch_size_tokens: 65536
model:
  class_path: matchcake_opt.modules.LanguageModel
  init_args:
    backbone:
      class_path: my_project.models.MyFermionicLM   # your model; vocab_size is filled in from the tokenizer
      init_args: {n_modes: 16, n_layers: 4}
optimizer:
  class_path: torch.optim.AdamW
  init_args: {lr: 1.0e-3, weight_decay: 0.0}
lr_scheduler:
  class_path: matchcake_opt.optim.WarmupWarmdownLR
  init_args: {warmup_steps: 40, warmdown_ratio: 0.65, final_lr_frac: 0.05}
trainer:
  devices: 1
  max_steps: 5000
  val_check_interval: 250
  limit_val_batches: 40
  precision: 32-true
  callbacks:
    - {class_path: matchcake_opt.callbacks.SamplingCallback, init_args: {every_n_steps: 1000}}
```

Commands:

```bash
matchcake-opt fit --config runs/pretrain_example.yaml
matchcake-opt fit --config runs/sft_example.yaml --model.init_args.init_from checkpoints/pretrain/last.ckpt
matchcake-opt chat-eval --config runs/sft_example.yaml --ckpt_path checkpoints/sft/last.ckpt --tasks "ARC-Easy|GSM8K"
```

### 3.6 Evaluation

- **Pretraining**: `val_loss`, `val_bpb` (a `torchmetrics.Metric` summing nats and bytes, synced across GPUs),
  samples from fixed prompts.
- **SFT / chat**: per-task accuracy.
  - Categorical (ARC-Easy, ARC-Challenge, MMLU): argmax over the answer-letter logits.
  - Generative (GSM8K, HumanEval): sampling with `LanguageModel.generate`; GSM8K checks the final number
    (calculator tool calls supported as in nanochat); HumanEval runs the code in a sandboxed subprocess
    (port of `execution.py`, Linux only).
  - Runs as a callback during SFT (every N steps, limited number of problems) and as the standalone
    `chat-eval` command.
- All results are written as JSON next to the checkpoints (same convention as `LightningPipeline`).

### 3.7 Package layout

New code follows the existing layout by component type. It goes into an existing file when it fits there
(e.g. new datasets in `datasets/`, new modules in `modules/`); new files are created only when needed:

```
src/matchcake_opt/
  tokenizers/      base.py, tiktoken_tokenizer.py, hf_tokenizer.py, nanochat_tokenizer.py, chat_template.py
  datasets/text/   sources.py (ParquetShards, HFTextDataset, TextFiles), packing.py, download.py
  tasks/           common.py, smoltalk.py, mmlu.py, gsm8k.py, arc.py, humaneval.py, custom_json.py
  datamodules/     pretrain_datamodule.py, sft_datamodule.py
  modules/         language_model.py, fermionic_lm.py (reference backbone)
  metrics/         bpb.py
  optim/           schedulers.py
  generation/      sampling.py, tools.py (calculator), execution.py (sandbox)
  callbacks/       sampling.py, chat_eval.py
  evaluation/      chat_eval.py
  cli.py           LightningCLI entrypoint (`matchcake-opt` console script)
runs/              example configs (pretrain, sft, chat-eval, tiny CPU smoke test)
```

### 3.8 New dependencies

`tiktoken`, `tokenizers`, `datasets`, `pyarrow`, `jsonargparse[signatures]` (for LightningCLI). Optional
extra `[wandb]`. nanochat code that is ported keeps its MIT copyright notice in the file header
(compatible with Apache-2.0).

### 3.9 Tests

Every PR adds unit tests that run on CPU in CI. Tests use a **tiny test-only backbone** under `tests/`
(nothing classical is shipped in `src/`). An end-to-end smoke test runs `fit` on a few hundred characters of
text for a few steps, then `chat-eval` on 2 problems.

## 4. Pull requests

Each PR targets `dev` on `Squynijos/MatchCake-Opt`, has tests, and passes black / isort / mypy.

| # | Branch | Content | Depends on |
|---|---|---|---|
| 0 | `docs/nanochat-pipeline-plan` | this plan | – |
| 1 | `feat/tokenizers` | `Tokenizer` interface, 3 adapters, chat template + loss mask, `token_bytes` | – |
| 2 | `feat/text-data` | text sources, ClimbMix shard download, BOS best-fit packing, `PretrainDataModule` (DDP/worker sharding, resume state) | 1 |
| 3 | `feat/language-model` | `LanguageModel`, bpb metric, `WarmupWarmdownLR`, generation | 1 |
| 4 | `feat/llm-cli` | LightningCLI entrypoint, config linking (vocab size, grad accumulation), sampling callback, example configs, smoke test → **end-to-end pretraining** | 2, 3 |
| 5 | `feat/fermionic-lm` | minimal fermionic reference backbone built with MatchCake (`modules/fermionic_lm.py`), used by the example configs | 3 |
| 6 | `feat/sft` | tasks + mixtures, `SFTDataModule`, `init_from` checkpoint | 4 |
| 7 | `feat/chat-eval` | categorical + generative eval, calculator tool, sandbox, callback + `chat-eval` command → **end-to-end SFT + eval** | 6 |
| 8 | `feat/rl` (later) | GRPO-style RL on GSM8K | 7 |
| 9 | `docs/llm-tutorial` | README section + notebook | 7 |

PRs 2 and 3 are independent and can be reviewed in parallel; PR 5 can start as soon as PR 3 is merged.

## 5. Review decisions

| # | Question | Decision |
|---|---|---|
| Q1 | Layout | Use the existing component folders; add to an existing file when it fits, new file only when needed. |
| Q2 | Example model | Ship a minimal fermionic reference backbone (PR 5). Tests still use a tiny test-only dummy. |
| Q3 | Precision / compile | Default `precision: 32-true`, no `torch.compile`; both switchable in the config. |
| Q4 | Optimizer | AdamW. The Muon port is dropped; any `torch.optim` optimizer can be set in the config. |
| Q5 | Default data | nanochat's ClimbMix parquet shards; HF datasets and local files as alternatives. |
| Q6 | AutoML | Out of scope for now. Keep it possible: components stay plain constructor arguments (no global state), hyperparameters stay in `init_args`, and the run returns its metrics as a dict like `LightningPipeline.run()`. |
| Q7 | Sequence length | Unknown. No hard limit in the code: `seq_len` is a config value everywhere, packing and evaluation never assume a minimum, and the example configs use small values (`seq_len: 256`, smoke test `seq_len: 32`). Long-prompt evals (MMLU, HumanEval) truncate prompts from the left to fit `seq_len` and report how many were truncated. |

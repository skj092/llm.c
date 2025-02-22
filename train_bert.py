"""
Reference code for training BERT from scratch for text classification.
Will save the model weights into files, to be read from C as initialization.

Example launch for binary classification:
python train_bert.py --num_iterations=50 --sequence_length=128 --dtype=bfloat16 --num_classes=2 --vocab_size=50257
"""

import os
import math
import glob
from contextlib import nullcontext
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for BERT

class NewGELU(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head

        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        q = self.query(x).view(B, T, self.n_head,
                               self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head,
                               self.head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = NewGELU()
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


@dataclass
class BertConfig:
    vocab_size: int = 50257  # Match GPT-2 vocab size used in Tiny Shakespeare data
    block_size: int = 128    # Max sequence length
    n_layer: int = 12        # Number of layers
    n_head: int = 12         # Number of attention heads
    n_embd: int = 768        # Embedding dimension


class BertForClassification(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(
            config.block_size, config.n_embd)
        self.layers = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.n_embd, num_classes)

        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0,
                                  std=0.02, generator=self.init_rng)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0,
                                  std=0.02, generator=self.init_rng)

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T = input_ids.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        x = self.dropout(x)

        mask = attention_mask[:, None, None,
                              :] if attention_mask is not None else None
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_f(x)

        cls_output = x[:, 0, :]
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, zero_stage):
        param_dict = {pn: p for pn, p in self.named_parameters()
                      if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print0(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        if zero_stage == 1:
            optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                                lr=learning_rate, betas=betas)
            optimizer.add_param_group(optim_groups[1])
        else:
            optimizer = torch.optim.AdamW(
                optim_groups, lr=learning_rate, betas=betas)
        return optimizer

# -----------------------------------------------------------------------------
# Distributed Data Loader for Classification


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        self.files = sorted(glob.glob(filename_pattern))
        assert len(
            self.files) > 0, f"No files match pattern {filename_pattern}"

        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(
            f"DataLoader: total number of samples: {ntok_total:,} across {len(self.files)} files")

        self.current_shard = None
        self.reset()

    def reset(self):
        if self.current_shard != 0 or self.current_shard is None:
            self.current_shard = 0
            self.data = _load_data_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B
        self.data = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        buf = self.data[self.current_position:self.current_position + B]
        input_ids = torch.tensor([item['input_ids']
                                 for item in buf], dtype=torch.long)
        attention_mask = torch.tensor(
            [item['attention_mask'] for item in buf], dtype=torch.long)
        labels = torch.tensor([item['label']
                              for item in buf], dtype=torch.long)

        self.current_position += B * self.num_processes
        if self.current_position + (B * self.num_processes) > len(self.data):
            self.advance()
        return input_ids, attention_mask, labels


def _peek_data_shard(filename):
    data = np.load(filename, allow_pickle=True)
    return len(data)


def _load_data_shard(filename):
    return np.load(filename, allow_pickle=True)

# -----------------------------------------------------------------------------
# Utilities for saving weights


def write_fp32(tensor, file):
    t = tensor.detach().cpu().to(torch.float32)
    b = t.numpy().tobytes()
    file.write(b)


def write_bf16(tensor, file):
    t = tensor.detach().cpu().to(torch.bfloat16)
    t = t.view(torch.int16)
    b = t.numpy().tobytes()
    file.write(b)


def write_tensors(model_tensors, L, file, dtype):
    write_fun = write_fp32 if dtype == "float32" else write_bf16
    write_fun(model_tensors["token_embedding.weight"], file)
    write_fun(model_tensors["position_embedding.weight"], file)
    for i in range(L):
        write_fun(model_tensors[f"layers.{i}.ln1.weight"], file)
        write_fun(model_tensors[f"layers.{i}.ln1.bias"], file)
        write_fun(model_tensors[f"layers.{i}.attn.query.weight"], file)
        write_fun(model_tensors[f"layers.{i}.attn.query.bias"], file)
        write_fun(model_tensors[f"layers.{i}.attn.key.weight"], file)
        write_fun(model_tensors[f"layers.{i}.attn.key.bias"], file)
        write_fun(model_tensors[f"layers.{i}.attn.value.weight"], file)
        write_fun(model_tensors[f"layers.{i}.attn.value.bias"], file)
        write_fun(model_tensors[f"layers.{i}.attn.proj.weight"], file)
        write_fun(model_tensors[f"layers.{i}.attn.proj.bias"], file)
        write_fun(model_tensors[f"layers.{i}.ln2.weight"], file)
        write_fun(model_tensors[f"layers.{i}.ln2.bias"], file)
        write_fun(model_tensors[f"layers.{i}.ff.fc1.weight"], file)
        write_fun(model_tensors[f"layers.{i}.ff.fc1.bias"], file)
        write_fun(model_tensors[f"layers.{i}.ff.fc2.weight"], file)
        write_fun(model_tensors[f"layers.{i}.ff.fc2.bias"], file)
    write_fun(model_tensors["ln_f.weight"], file)
    write_fun(model_tensors["ln_f.bias"], file)
    write_fun(model_tensors["classifier.weight"], file)
    write_fun(model_tensors["classifier.bias"], file)


def write_model(model, filename, dtype):
    assert dtype in {"float32", "bfloat16"}
    version = {"float32": 3, "bfloat16": 5}[dtype]
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240326  # magic
    header[1] = version
    header[2] = model.config.block_size
    header[3] = model.config.vocab_size
    header[4] = model.config.n_layer
    header[5] = model.config.n_head
    header[6] = model.config.n_embd

    params = {name: param.cpu() for name, param in model.named_parameters()}
    with open(filename, "wb") as file:
        file.write(header.numpy().tobytes())
        write_tensors(params, model.config.n_layer, file, dtype)
    print(f"wrote {filename}")


# -----------------------------------------------------------------------------
# Main
def print0(*args, **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


if __name__ == "__main__":
    import time
    import argparse
    print0(f"Running PyTorch {torch.version.__version__}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_bin", type=str,
                        default="./dev/data/tinyshakespeare/tiny_shakespeare_train_class.npy", help="input .npy for training")
    parser.add_argument("--input_val_bin", type=str,
                        default="", help="input .npy for validation")
    parser.add_argument("--output_dir", type=str, default="",
                        help="output directory for logs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument("--total_batch_size", type=int, default=512)
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_loss_every", type=int, default=5)
    parser.add_argument("--val_max_steps", type=int, default=20)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--write_tensors", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classification classes")
    parser.add_argument("--n_layer", type=int, default=12,
                        help="number of layers")
    parser.add_argument("--n_head", type=int, default=12,
                        help="number of attention heads")
    parser.add_argument("--n_embd", type=int, default=768,
                        help="embedding dimension")
    parser.add_argument("--vocab_size", type=int,
                        default=50257, help="vocabulary size")
    args = parser.parse_args()

    B, T = args.batch_size, args.sequence_length
    assert args.dtype in {"float32", "bfloat16"}

    # DDP setup
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        zero_stage = 1
    else:
        ddp_rank = 0
        ddp_world_size = 1
        master_process = True
        zero_stage = 0
        device = args.device if args.device else (
            "cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    tokens_per_fwdbwd = B * T * ddp_world_size
    grad_accum_steps = max(1, args.total_batch_size // tokens_per_fwdbwd)
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    ptdtype = {'float32': torch.float32,
               'bfloat16': torch.bfloat16}[args.dtype]
    ctx = torch.amp.autocast(
        device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    config = BertConfig(
        vocab_size=args.vocab_size,
        block_size=args.sequence_length,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd
    )
    model = BertForClassification(config, args.num_classes)
    model.train()
    model.to(device)

    train_loader = DistributedDataLoader(
        args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = DistributedDataLoader(
        args.input_val_bin, B, T, ddp_rank, ddp_world_size) if args.input_val_bin else None

    if master_process and args.write_tensors:
        write_model(
            model, f"bert_scratch_{args.n_layer}L_{args.n_embd}D.bin", dtype=args.dtype)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay,
                                               learning_rate=args.learning_rate, betas=(
                                                   0.9, 0.95),
                                               device_type=device, zero_stage=zero_stage)

    def get_lr(it):
        min_lr = args.learning_rate * 0.1
        if it < args.warmup_iters:
            return args.learning_rate * (it + 1) / args.warmup_iters
        if it > args.num_iterations:
            return min_lr
        decay_ratio = (it - args.warmup_iters) / \
            (args.num_iterations - args.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (args.learning_rate - min_lr)

    logfile = None
    if args.output_dir and master_process:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "main.log")
        with open(logfile, "w") as f:
            pass

    timings = []
    for step in range(args.num_iterations + 1):
        t0 = time.time()
        last_step = (step == args.num_iterations)

        if args.val_loss_every > 0 and (step % args.val_loss_every == 0 or last_step) and val_loader:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(args.val_max_steps):
                    input_ids, attention_mask, labels = val_loader.next_batch()
                    input_ids, attention_mask, labels = input_ids.to(
                        device), attention_mask.to(device), labels.to(device)
                    _, loss = model(input_ids, attention_mask, labels)
                    val_loss += loss.item()
                val_loss /= args.val_max_steps
            print0(f"step {step}: val loss {val_loss:.6f}")
            if master_process and logfile:
                with open(logfile, "a") as f:
                    f.write(f"s:{step} val:{val_loss:.6f}\n")

        if last_step:
            break

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.tensor(0.0, device=device)
        for micro_step in range(grad_accum_steps):
            input_ids, attention_mask, labels = train_loader.next_batch()
            input_ids, attention_mask, labels = input_ids.to(
                device), attention_mask.to(device), labels.to(device)
            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == grad_accum_steps - 1)
            with ctx:
                _, loss = model(input_ids, attention_mask, labels)
                loss = loss / grad_accum_steps
                loss_accum += loss
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        lossf = loss_accum.item()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        tokens_per_second = grad_accum_steps * \
            ddp_world_size * B * T / (t1 - t0)
        print0(f"step {step+1:4d}/{args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | {(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s")
        if master_process and logfile:
            with open(logfile, "a") as f:
                f.write(f"s:{step} trl:{lossf:.6f}\n")

        if step > 0:
            timings.append(t1 - t0)

    if timings:
        avg_time = np.mean(timings[-20:]) * 1000
        print0(f"final {len(timings)} iters avg: {avg_time:.3f} ms")
    if device_type == "cuda":
        print0(
            f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    if ddp:
        destroy_process_group()

"""
Reference code for training BERT for text classification with pretrained weights.
Will save the model weights into files, to be read from C as initialization.

Example launch:
python train_bert.py --num_iterations=50 --sequence_length=512 --dtype=float32 --num_classes=4 --vocab_size=30522
"""

import torch.utils.data as data
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers import BertForSequenceClassification
import torch.distributed as dist
from dataclasses import dataclass
from bert import CustomBertForSequenceClassification  # Assuming bert.py has your custom model
from contextlib import nullcontext

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for BERT
@dataclass
class BertConfig:
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0

# -----------------------------------------------------------------------------
# Distributed Data Loader for Classification
class DistributedDataLoader:
    def __init__(self, data_prefix, B, T, process_rank, num_processes, vocab_size):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B  # Batch size
        self.T = T  # Sequence length
        self.vocab_size = vocab_size

        # Load binary data files
        input_ids = np.fromfile(f"{data_prefix}_input_ids.bin", dtype=np.int32)
        attention_masks = np.fromfile(f"{data_prefix}_attention_masks.bin", dtype=np.int32)
        labels = np.fromfile(f"{data_prefix}_labels.bin", dtype=np.int64)

        # Ensure consistent length
        num_samples = min(len(input_ids) // T, len(attention_masks) // T, len(labels))
        print(f"📌 Loading {num_samples} samples from {data_prefix}")

        # Trim to ensure valid shape
        input_ids = input_ids[:num_samples * T].reshape(num_samples, T)
        attention_masks = attention_masks[:num_samples * T].reshape(num_samples, T)
        labels = labels[:num_samples]

        # Validate input_ids against vocab_size
        max_token_id = input_ids.max()
        if max_token_id >= vocab_size:
            raise ValueError(f"🚨 Input IDs contain token {max_token_id} which exceeds vocab_size={vocab_size}. "
                             f"Adjust --vocab_size to match the tokenizer used for the dataset.")

        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.total_samples = num_samples

        self.current_position = process_rank * B

    def reset(self):
        self.current_position = self.process_rank * self.B

    def advance(self):
        self.current_position += self.B * self.num_processes
        if self.current_position >= self.total_samples:
            self.reset()

    def next_batch(self):
        start, end = self.current_position, self.current_position + self.B
        if end > self.total_samples:
            self.advance()
            start, end = self.current_position, self.current_position + self.B

        x = torch.tensor(self.input_ids[start:end], dtype=torch.long)
        attn = torch.tensor(self.attention_masks[start:end], dtype=torch.long)
        y = torch.tensor(self.labels[start:end], dtype=torch.long)

        self.advance()
        return x, attn, y

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

def write_tensors(model_tensors, config, file, dtype):
    write_fun = write_fp32 if dtype == "float32" else write_bf16
    write_fun(model_tensors["embeddings.word_embeddings.weight"], file)
    write_fun(model_tensors["embeddings.position_embeddings.weight"], file)
    write_fun(model_tensors["embeddings.token_type_embeddings.weight"], file)
    write_fun(model_tensors["embeddings.LayerNorm.weight"], file)
    write_fun(model_tensors["embeddings.LayerNorm.bias"], file)
    for i in range(config.num_hidden_layers):
        prefix = f"encoder.{i}"
        write_fun(model_tensors[f"{prefix}.attention.self.query.weight"], file)
        write_fun(model_tensors[f"{prefix}.attention.self.query.bias"], file)
        write_fun(model_tensors[f"{prefix}.attention.self.key.weight"], file)
        write_fun(model_tensors[f"{prefix}.attention.self.key.bias"], file)
        write_fun(model_tensors[f"{prefix}.attention.self.value.weight"], file)
        write_fun(model_tensors[f"{prefix}.attention.self.value.bias"], file)
        write_fun(model_tensors[f"{prefix}.attention.output.dense.weight"], file)
        write_fun(model_tensors[f"{prefix}.attention.output.dense.bias"], file)
        write_fun(model_tensors[f"{prefix}.attention.output.LayerNorm.weight"], file)
        write_fun(model_tensors[f"{prefix}.attention.output.LayerNorm.bias"], file)
        write_fun(model_tensors[f"{prefix}.intermediate.dense.weight"], file)
        write_fun(model_tensors[f"{prefix}.intermediate.dense.bias"], file)
        write_fun(model_tensors[f"{prefix}.output.dense.weight"], file)
        write_fun(model_tensors[f"{prefix}.output.dense.bias"], file)
        write_fun(model_tensors[f"{prefix}.output.LayerNorm.weight"], file)
        write_fun(model_tensors[f"{prefix}.output.LayerNorm.bias"], file)
    write_fun(model_tensors["pooler.weight"], file)
    write_fun(model_tensors["pooler.bias"], file)
    write_fun(model_tensors["classifier.weight"], file)
    write_fun(model_tensors["classifier.bias"], file)

def write_model(model, filename, dtype):
    assert dtype in {"float32", "bfloat16"}
    version = {"float32": 3, "bfloat16": 5}[dtype]
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240326  # magic
    header[1] = version
    header[2] = model.config.max_position_embeddings
    header[3] = model.config.vocab_size
    header[4] = model.config.num_hidden_layers
    header[5] = model.config.num_attention_heads
    header[6] = model.config.hidden_size

    params = {name: param.cpu() for name, param in model.named_parameters()}
    with open(filename, "wb") as file:
        file.write(header.numpy().tobytes())
        write_tensors(params, model.config, file, dtype)
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
    parser.add_argument("--data_dir", type=str, default="./dev/data/", help="Directory containing .bin files")
    parser.add_argument("--val_data_dir", type=str, default="./dev/data/", help="Validation data directory")
    parser.add_argument("--input_bin", type=str, default="./dev/data/", help="input .npy for training")
    parser.add_argument("--input_val_bin", type=str, default="./dev/data/", help="input .npy for validation")
    parser.add_argument("--output_dir", type=str, default="", help="output directory for logs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--total_batch_size", type=int, default=4)
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_loss_every", type=int, default=5)
    parser.add_argument("--val_max_steps", type=int, default=20)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--write_tensors", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=4, help="number of classification classes")
    parser.add_argument("--n_layer", type=int, default=12, help="number of layers")
    parser.add_argument("--n_head", type=int, default=12, help="number of attention heads")
    parser.add_argument("--n_embd", type=int, default=768, help="embedding dimension")
    parser.add_argument("--vocab_size", type=int, default=30522, help="vocabulary size")
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
    else:
        ddp_rank = 0
        ddp_world_size = 1
        master_process = True
        device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    tokens_per_fwdbwd = B * T * ddp_world_size
    grad_accum_steps = max(1, args.total_batch_size // tokens_per_fwdbwd)
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Configure model
    config = BertConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.n_embd,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        intermediate_size=args.n_embd * 4,
        max_position_embeddings=args.sequence_length
    )

    # Initialize model
    model = CustomBertForSequenceClassification(config, args.num_classes)
    pretrained_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=args.num_classes)
    model.load_pretrained_weights(pretrained_model)

    model.train()
    model.to(device)

    # Load data
    train_loader = DistributedDataLoader("./dev/data/ag-news/ag_news_train", B, T, ddp_rank, ddp_world_size, args.vocab_size)
    val_loader = DistributedDataLoader("./dev/data/ag-news/ag_news_test", B, T, ddp_rank, ddp_world_size, args.vocab_size)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )

    def get_lr(it):
        min_warmup = 1000
        warmup_iters = max(args.warmup_iters, min_warmup)
        min_lr = args.learning_rate * 0.1
        if it < warmup_iters:
            return args.learning_rate * (it + 1) / warmup_iters
        if it > args.num_iterations:
            return min_lr
        decay_ratio = (it - warmup_iters) / (args.num_iterations - warmup_iters)
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
                    input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                    _, loss = model(input_ids, attention_mask, labels=labels)
                    val_loss += loss.item()
                val_loss /= args.val_max_steps
            print0(f"step {step}: val loss {val_loss:.6f}")
            if master_process and logfile:
                with open(logfile, "a") as f:
                    f.write(f"s:{step} val:{val_loss:.6f}\n")

        if last_step:
            if master_process and args.write_tensors:
                write_model(model, f"bert_scratch_{args.n_layer}L_{args.n_embd}D.bin", dtype=args.dtype)
            break

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.tensor(0.0, device=device)
        for micro_step in range(grad_accum_steps):
            input_ids, attention_mask, labels = train_loader.next_batch()
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            # Explicitly set token_type_ids and debug
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=device)
            print0(f"Step {step}, Micro-step {micro_step}: input_ids min={input_ids.min().item()}, max={input_ids.max().item()}")
            print0(f"Step {step}, Micro-step {micro_step}: token_type_ids min={token_type_ids.min().item()}, max={token_type_ids.max().item()}")

            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with ctx:
                _, loss = model(input_ids, attention_mask, token_type_ids=token_type_ids, labels=labels)
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        lossf = loss_accum.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        tokens_per_second = grad_accum_steps * ddp_world_size * B * T / (t1 - t0)
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
        print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    if ddp:
        destroy_process_group()

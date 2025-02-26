from bert_dev import BertEmbeddings, generate_random_input, bert_base, config
import torch.nn.functional as F
import pdb
from transformers import BertConfig, BertModel
import torch
import numpy as np
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertSdpaSelfAttention
import math


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5  # Scaling factor for attention scores

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, query, key, value, dropout_p=0.0) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value

    def forward(self, x):
        batch_size, seq_length, hidden_size = x.shape

        # Project query, key, value
        q = self.query(x).view(batch_size, seq_length,
                               self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_length,
                             self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_length,
                               self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = self.scaled_dot_product_attention(q, k, v)

        # Apply attention to values
        context = attn_scores.transpose(1, 2).contiguous().view(
            batch_size, seq_length, hidden_size)

        return (context, )


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size=768, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual):
        x = self.dense(x)
        x = self.dropout(x)
        return self.LayerNorm(x + residual)


class BertAttention(nn.Module):
    def __init__(self, config, hidden_size=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.self = BertSelfAttention(hidden_size, num_heads, dropout)
        self.output = BertSelfOutput(hidden_size, dropout)

    def forward(self, x):
        attn_output = self.self(x)
        output = self.output(attn_output, x)  # Add residual connection
        return output

    def load_from_pretrained(self):
        config = BertConfig()
        emb = BertAttention(config).self
        sd = emb.state_dict()

        hf_sd = bert_base.encoder.layer[0].attention.self.state_dict()

        for key, hf_val in hf_sd.items():
            print(f"Copying {key}")
            assert hf_sd[key].shape == sd[
                key].shape, f"Shape mismatch for {key}"

            with torch.no_grad():
                sd[key].copy_(hf_sd[key])
        return emb


input = torch.rand(2, 128, 768)

# custom model
sa_m = BertAttention(config).load_from_pretrained()
sa_m.eval()
out1 = sa_m(input)[0]

# hf  model
sa = bert_base.encoder.layer[0].attention.self
out2 = sa(input)[0]

assert torch.allclose(out1, out2, atol=1e-6), "❌ self attention  Mismatch!"

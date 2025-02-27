"""In this script I'll try to reproduce the bert from scratch, load pretrained model weight and get same output as pretrained_model"""

from transformers import BertModel
import torch
from dataclasses import dataclass
import numpy as np
import torch.nn as nn
import math


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


@dataclass
class BertConfig:
    is_decoder: bool = False
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
    type_vocab_size: int = 2


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(p=0.1)

        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

    def forward(self, input_ids, token_type_ids=None):
        batch_size, seq_length = input_ids.size()

        # Word Embeddings
        word_embed = self.word_embeddings(input_ids)

        # Position Embeddings
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)
        position_embd = self.position_embeddings(position_ids)

        # Token Type Embeddings
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        token_type_embd = self.token_type_embeddings(token_type_ids)

        # Combine all embeddings
        embeddings = word_embed + position_embd + token_type_embd

        # Apply LayerNorm and Dropout
        # print('before layer norm')
        # print(embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


def generate_random_input(batch_size=2, seq_length=128, vocab_size=30522):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    return input_ids, attention_mask, token_type_ids


class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5  # Scaling factor for attention scores

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(
        self, query, key, value, dropout_p=0.0
    ) -> torch.Tensor:
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
        q = (
            self.query(x)
            .view(batch_size, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.key(x)
            .view(batch_size, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value(x)
            .view(batch_size, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Compute attention scores
        attn_scores = self.scaled_dot_product_attention(q, k, v)

        # Apply attention to values
        context = (
            attn_scores.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, hidden_size)
        )

        return (context,)


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, xb):
        xb = self.dense(xb)
        out = self.intermediate_act_fn(xb)
        return out


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, hidden_size=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.self = BertSelfAttention(hidden_size, num_heads, dropout)
        self.output = BertSelfOutput(config)

    def forward(self, x):
        self_outputs = self.self(x)
        attention_output = self.output(self_outputs[0], x)
        outputs = (attention_output,) + self_outputs[1:]

        return outputs


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_state):
        attn_out = self.attention(hidden_state)
        extra_outputs = attn_out[1:]
        intermediate = self.intermediate(attn_out[0])
        layer_output = self.output(intermediate, attn_out[0])
        outputs = (layer_output,) + extra_outputs

        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList(BertLayer(config) for _ in range(12))

    def forward(self, hidden_state):
        for l in self.layer:
            hidden_state = l(hidden_state)[0]
        return hidden_state


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModelCustom(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def load_from_pretrained(self):
        hf_sd = bert_base.state_dict()

        for k in hf_sd.keys():
            self.state_dict()[k].copy_(hf_sd[k])
        return self

    def forward(self, input_ids):
        out = self.embeddings(input_ids)
        out = self.encoder(out)
        p_out = self.pooler(out)
        return out, p_out


if __name__ == "__main__":
    config = BertConfig()
    input_ids, attention_mask, token_type_ids = generate_random_input()

    # Load base BERT model
    bert_base = BertModel.from_pretrained("bert-base-uncased")
    bert_base.eval()
    out1 = bert_base(input_ids)

    # Custom Model
    model = BertModelCustom(config).load_from_pretrained()
    model.eval()
    out2 = model(input_ids)

    # validate output
    for i in range(len(out1)):
        assert torch.allclose(
            out1[i], out2[i], atol=1e-5), "❌ out Layer  Mismatch!"

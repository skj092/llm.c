'''In this script I'll try to reproduce the bert from scratch, load pretrained model weight and get same output as pretrained_model'''

from torch.nn.functional import embedding
from transformers import BertConfig, BertModel
import code
from transformers import BertForSequenceClassification, BertModel
import torch
from dataclasses import dataclass
import numpy as np
import torch.nn as nn
from transformers.modeling_utils import dtype_byte_size
import math
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertSdpaSelfAttention


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


config = BertConfig()
num_classes = 2

# # Load base BERT model first, then create classification model
bert_base = BertModel.from_pretrained('bert-base-uncased')
bert_base.eval()
# pretrained_model = BertForSequenceClassification.from_pretrained(
#     'bert-base-uncased', num_labels=num_classes)


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(p=0.1)

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def load_from_pretrained(self):
        config = BertConfig()
        emb = BertEmbeddings(config)
        sd = emb.state_dict()

        hf_model = BertModel.from_pretrained('bert-base-uncased')
        hf_sd = hf_model.embeddings.state_dict()

        assert sd.keys() == hf_sd.keys(
        ), f"mismatch keys {len(sd.keys())} != {len(hf_sd.keys())}"

        for k in hf_sd.keys():
            print(f"copying weight of {k}")
            assert hf_sd[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(hf_sd[k])
        return emb

    def forward(self, input_ids, token_type_ids=None):
        batch_size, seq_length = input_ids.size()

        # Word Embeddings
        word_embed = self.word_embeddings(input_ids)

        # Position Embeddings
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
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


class BertForSequenceClassificationCustom(nn.Module):
    def __init__(self, config, num_classes):
        self.config = config
        self.num_classes = num_classes
        self.embeddings = BertEmbeddings(config)
        self.encoder = nn.ModuleList(BertLayer(config)
                                     for _ in range(config.num_hidden_layers))
        self.pooler = BertPooler(config)


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


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def load_from_pretrained(self):
        config = BertConfig()
        emb = BertAttention(config)
        sd = emb.state_dict()

        hf_sd = bert_base.encoder.layer[0].output.state_dict()

        for key in hf_sd.keys():
            print(f"Copying {key}")
            assert hf_sd[key].shape == sd[
                key].shape, f"Shape mismatch for {key}"

            with torch.no_grad():
                sd[key].copy_(hf_sd[key])
        return emb


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def load_from_pretrained(self):
        hf_bi = bert_base.encoder.layer[0].intermediate
        hf_bi_sd = hf_bi.state_dict()
        for k in hf_bi_sd.keys():
            self.state_dict()[k].copy_(hf_bi_sd[k])
        return self

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

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def load_from_pretrained(self):
        hf_m = bert_base.encoder.layer[0].output
        hf_m_sd = hf_m.state_dict()

        for k in hf_m_sd.keys():
            print(f"Copying {k}")
            self.state_dict()[k].copy_(hf_m_sd[k])
        return self


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def load_from_pretrained(self):
        config = BertConfig()
        emb = BertAttention(config)
        sd = emb.state_dict()

        hf_sd = bert_base.encoder.layer[0].attention.state_dict()

        for key in hf_sd.keys():
            print(f"Copying {key}")
            assert hf_sd[key].shape == sd[
                key].shape, f"Shape mismatch for {key}"

            with torch.no_grad():
                sd[key].copy_(hf_sd[key])
        return emb


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

    def load_from_pretrained(self):
        config = BertConfig()
        emb = BertAttention(config)
        sd = emb.state_dict()

        hf_sd = bert_base.encoder.layer[0].attention.state_dict()

        for key in hf_sd.keys():
            print(f"Copying {key}")
            assert hf_sd[key].shape == sd[
                key].shape, f"Shape mismatch for {key}"

            with torch.no_grad():
                sd[key].copy_(hf_sd[key])
        return emb


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def load_from_pretrained(self):
        hf_enc = bert_base.encoder.layer[0]
        hf_enc_sd = hf_enc.state_dict()

        for k in hf_enc_sd.keys():
            self.state_dict()[k].copy_(hf_enc_sd[k])
        return self

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

    def load_from_pretrained(self):
        hf_enc = bert_base.encoder
        hf_enc_sd = hf_enc.state_dict()

        for k in hf_enc_sd.keys():
            self.state_dict()[k].copy_(hf_enc_sd[k])
        return self

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
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


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
    type_vocab_size: int = 2


config = BertConfig()
num_classes = 2

# # Load base BERT model first, then create classification model
bert_base = BertModel.from_pretrained('bert-base-uncased')
bert_base.eval()
# pretrained_model = BertForSequenceClassification.from_pretrained(
#     'bert-base-uncased', num_labels=num_classes)


class BertLayer(nn.Module):
    pass


class BertPooler(nn.Module):
    pass


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

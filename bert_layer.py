from transformers import BertModel
from torch.autograd import forward_ad
from bert_dev import BertEmbeddings, BertAttention, config, BertSelfOutput, bert_base
import pdb
import torch
import numpy as np
import torch.nn as nn


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


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


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def load_from_pretrained(self):
        hf_enc = bert_base.encoder.layer[0].intermediate
        hf_enc_sd = hf_enc.state_dict()

        sm = BertEncoder(config).intermediate
        sm_sd = sm.state_dict()

        for k in hf_enc_sd.keys():
            assert sm_sd[k].shape == hf_enc_sd[k].shape, "key not matching"
            sm_sd[k] = hf_enc_sd[k]
        return sm

    def forward(self):
        pass


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


input = torch.rand(2, 768, 3072)
hidden_state = torch.rand(768, 768)
# hidden_state = torch.rand(768, 129)

# dense: 3072, 768
# layernorm(dropout(dense(hidden_state)) + input_tensor)


#
# # custom model
bi_s= BertOutput(config).load_from_pretrained()
bi_s.eval()
out1= bi_s(input, hidden_state)
# #
# # hf  model
sa= bert_base.encoder.layer[0].output
sa.eval()
out2 = sa.forward(input, hidden_state)


assert torch.allclose(out1, out2, atol=1e-5), "❌ self attention  Mismatch!"

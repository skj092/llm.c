import pdb
from transformers import BertModel
from bert_dev import (BertEmbeddings, BertAttention, config,
                      BertSelfOutput, bert_base, BertIntermediate, BertOutput)
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
        print(hidden_state)
        return hidden_state


hidden_state = torch.rand(2, 768, 768)
# # custom model
bi_s = BertEncoder(config).load_from_pretrained()
bi_s.eval()
out1 = bi_s(hidden_state)
# #
# # hf  model
sa = bert_base.encoder
sa.eval()
out2 = sa.forward(hidden_state).last_hidden_state
# import pdb; pdb.set_trace()

assert torch.allclose(
    out1, out2, atol=1e-4), "❌ self attention  Mismatch!"

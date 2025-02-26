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
        self.output = BertSelfOutput(config)

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


input = torch.rand(2, 129, 768)

#
# # custom model
bi_s = BertIntermediate(config).load_from_pretrained()
out1 = bi_s(input)[0]
#
# # hf  model
sa = bert_base.encoder.layer[0].intermediate
out2 = sa.forward(input)[0]

assert torch.allclose(out1, out2, atol=1e-5), "❌ self attention  Mismatch!"

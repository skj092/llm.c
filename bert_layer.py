from os import confstr_names
import pdb
import torch
import numpy as np
import torch.nn as nn
from transformers import BertModel
from bert_dev import (config, BertAttention, BertOutput,
                      BertIntermediate, bert_base, BertEncoder, BertLayer, BertEmbeddings, BertPooler)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


class BertModelCustom(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def load_from_pretrained(self):
        hf_sd = bert_base.state_dict()

        for k in hf_sd.keys():
            print(f"Copying {k}")
            self.state_dict()[k].copy_(hf_sd[k])
        return self

    def forward(self):
        pass


hidden_state = torch.rand(2, 128, 768)

model = BertModelCustom(config).load_from_pretrained()
model.eval()
pdb.set_trace()

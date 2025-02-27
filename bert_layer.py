import pdb
import torch
import numpy as np
import torch.nn as nn
from transformers import BertModel
from bert_dev import (config, BertAttention, BertOutput,
                      BertIntermediate, bert_base, BertEncoder, BertLayer, BertEmbeddings, BertPooler, generate_random_input)


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
            # print(f"Copying {k}")
            self.state_dict()[k].copy_(hf_sd[k])
        return self

    def forward(self, input_ids):
        out = self.embeddings(input_ids)
        out = self.encoder(out)
        p_out = self.pooler(out)
        return out, p_out


input_ids, attention_mask, token_type_ids = generate_random_input()

model = BertModelCustom(config).load_from_pretrained()
model.eval()
out1 = model(input_ids)

out2 = bert_base(input_ids)


# for a, b in zip(out1, out2):
for i in range(len(out1)):
    assert torch.allclose(out1[i], out2[i], atol=1e-5), f"❌ out Layer  Mismatch!"
    print(f"matched")

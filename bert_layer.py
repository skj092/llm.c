from bert_dev import BertEmbeddings, generate_random_input, bert_base, config
import code
from transformers import BertConfig, BertModel
import torch
import numpy as np
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertSdpaSelfAttention


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def map_hf_keys_to_custom(hf_sd):
    key_map = {}
    for hf_key in hf_sd.keys():
        # Remove 'attention.self.' from HF keys and insert 'attention.attention.'
        new_key = hf_key.replace("query", "attention.attention.query") \
                        .replace("key", "attention.attention.key") \
                        .replace("value", "attention.attention.value")
        key_map[hf_key] = new_key
    return key_map


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertSdpaSelfAttention(config)

    def forward(self, xb):
        return self.attention(xb)


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)

    def load_from_pretrained(self):
        config = BertConfig()
        emb = BertLayer(config)
        sd = emb.state_dict()

        hf_sd = bert_base.encoder.layer[0].attention.self.state_dict()
        key_map = map_hf_keys_to_custom(hf_sd)

        for hf_key, model_key in key_map.items():
            print(f"Copying {hf_key} -> {model_key}")
            assert hf_sd[hf_key].shape == sd[model_key].shape, f"Shape mismatch for {hf_key}"

            with torch.no_grad():
                sd[model_key].copy_(hf_sd[hf_key])

    def forward(self, xb):
        return self.attention(xb)


input_ids, _, token_type_ids = generate_random_input()
emb = BertEmbeddings(config).load_from_pretrained()
emb.eval()


temp1 = bert_base.embeddings(
    input_ids=input_ids, token_type_ids=token_type_ids)
temp2 = emb(input_ids, token_type_ids)
assert torch.allclose(temp1, temp2, atol=1e-6), "❌ Word Embeddings Mismatch!"

sa_m = BertLayer(config)
sa_m.load_from_pretrained()
sa_m.eval()
# code.interact(local=locals())

sa = bert_base.encoder.layer[0].attention.self
out1 = sa(temp1)[0]
out2 = sa_m(temp1)[0]
print(out1)
print('*'*50)
print(out2)

assert torch.allclose(out1, out2, atol=1e-6), "❌ self attention  Mismatch!"
print("✅ Word Embeddings Match! 🎉")

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from transformers import BertForSequenceClassification, BertModel
import numpy as np
import os
import inspect
from torch.distributed.optim import ZeroRedundancyOptimizer


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


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


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

# Use exact same GELU as transformers


class BertGELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.scale = math.sqrt(self.attention_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / self.scale

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / self.scale

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = BertGELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class CustomBertForSequenceClassification(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config

        self.embeddings = nn.ModuleDict({
            'word_embeddings': nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id),
            'position_embeddings': nn.Embedding(config.max_position_embeddings, config.hidden_size),
            'token_type_embeddings': nn.Embedding(2, config.hidden_size),
            'LayerNorm': BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        })
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.encoder = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def load_pretrained_weights(self, pretrained_model):
        # Load embeddings
        self.embeddings['word_embeddings'].weight.data.copy_(
            pretrained_model.bert.embeddings.word_embeddings.weight.data)
        self.embeddings['position_embeddings'].weight.data.copy_(
            pretrained_model.bert.embeddings.position_embeddings.weight.data)
        self.embeddings['token_type_embeddings'].weight.data.copy_(
            pretrained_model.bert.embeddings.token_type_embeddings.weight.data)
        self.embeddings['LayerNorm'].weight.data.copy_(
            pretrained_model.bert.embeddings.LayerNorm.weight.data)
        self.embeddings['LayerNorm'].bias.data.copy_(
            pretrained_model.bert.embeddings.LayerNorm.bias.data)

        # Load encoder layers
        for i, layer in enumerate(self.encoder):
            layer.attention.self.query.weight.data.copy_(
                pretrained_model.bert.encoder.layer[i].attention.self.query.weight.data)
            layer.attention.self.query.bias.data.copy_(
                pretrained_model.bert.encoder.layer[i].attention.self.query.bias.data)
            layer.attention.self.key.weight.data.copy_(
                pretrained_model.bert.encoder.layer[i].attention.self.key.weight.data)
            layer.attention.self.key.bias.data.copy_(
                pretrained_model.bert.encoder.layer[i].attention.self.key.bias.data)
            layer.attention.self.value.weight.data.copy_(
                pretrained_model.bert.encoder.layer[i].attention.self.value.weight.data)
            layer.attention.self.value.bias.data.copy_(
                pretrained_model.bert.encoder.layer[i].attention.self.value.bias.data)
            layer.attention.output.dense.weight.data.copy_(
                pretrained_model.bert.encoder.layer[i].attention.output.dense.weight.data)
            layer.attention.output.dense.bias.data.copy_(
                pretrained_model.bert.encoder.layer[i].attention.output.dense.bias.data)
            layer.attention.output.LayerNorm.weight.data.copy_(
                pretrained_model.bert.encoder.layer[i].attention.output.LayerNorm.weight.data)
            layer.attention.output.LayerNorm.bias.data.copy_(
                pretrained_model.bert.encoder.layer[i].attention.output.LayerNorm.bias.data)
            layer.intermediate.dense.weight.data.copy_(
                pretrained_model.bert.encoder.layer[i].intermediate.dense.weight.data)
            layer.intermediate.dense.bias.data.copy_(
                pretrained_model.bert.encoder.layer[i].intermediate.dense.bias.data)
            layer.output.dense.weight.data.copy_(
                pretrained_model.bert.encoder.layer[i].output.dense.weight.data)
            layer.output.dense.bias.data.copy_(
                pretrained_model.bert.encoder.layer[i].output.dense.bias.data)
            layer.output.LayerNorm.weight.data.copy_(
                pretrained_model.bert.encoder.layer[i].output.LayerNorm.weight.data)
            layer.output.LayerNorm.bias.data.copy_(
                pretrained_model.bert.encoder.layer[i].output.LayerNorm.bias.data)

        # Load pooler and classifier
        self.pooler.weight.data.copy_(
            pretrained_model.bert.pooler.dense.weight.data)
        self.pooler.bias.data.copy_(
            pretrained_model.bert.pooler.dense.bias.data)
        self.classifier.weight.data.copy_(
            pretrained_model.classifier.weight.data)
        self.classifier.bias.data.copy_(pretrained_model.classifier.bias.data)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, return_layer_outputs=False):
        batch_size, seq_length = input_ids.size()

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_embeddings = self.embeddings['word_embeddings'](input_ids)
        position_embeddings = self.embeddings['position_embeddings'](
            position_ids)
        token_type_embeddings = self.embeddings['token_type_embeddings'](
            token_type_ids)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.embeddings['LayerNorm'](embeddings)
        embeddings = self.dropout(embeddings)

        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (
                1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None

        hidden_states = embeddings
        layer_outputs = [hidden_states] if return_layer_outputs else None

        for layer in self.encoder:
            hidden_states = layer(hidden_states, extended_attention_mask)
            if return_layer_outputs:
                layer_outputs.append(hidden_states)

        pooled_output = self.pooler(hidden_states[:, 0])
        pooled_output = self.pooler_activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        if return_layer_outputs:
            return logits, loss, layer_outputs
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, zero_stage):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print0(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(
            torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        print0(f"using fused AdamW: {use_fused}")
        if zero_stage == 1:
            print0("using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                                lr=learning_rate, betas=betas, fused=use_fused)
            optimizer.add_param_group(optim_groups[1])
        else:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(
                optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        return optimizer


def generate_random_input(batch_size=2, seq_length=128, vocab_size=30522):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    return input_ids, attention_mask, token_type_ids


def compare_layer_outputs(pretrained_model, custom_model, input_ids, attention_mask, token_type_ids):
    pretrained_model.eval()
    custom_model.eval()

    # Ensure dropout is disabled
    for module in pretrained_model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    for module in custom_model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

    with torch.no_grad():
        pretrained_outputs = pretrained_model.bert(input_ids, attention_mask, token_type_ids,
                                                   output_hidden_states=True)
        pretrained_layer_outputs = pretrained_outputs.hidden_states

        _, _, custom_layer_outputs = custom_model(input_ids, attention_mask, token_type_ids,
                                                  return_layer_outputs=True)

        print("\nLayer-by-layer comparison:")
        for i, (pretrained_out, custom_out) in enumerate(zip(pretrained_layer_outputs, custom_layer_outputs)):
            max_diff = torch.abs(pretrained_out - custom_out).max()
            mean_diff = torch.abs(pretrained_out - custom_out).mean()
            similarity = F.cosine_similarity(
                pretrained_out.view(-1), custom_out.view(-1), dim=0)
            print(f"Layer {i} (shape: {pretrained_out.shape}):")
            print(f"  Max difference: {max_diff.item():.8f}")
            print(f"  Mean difference: {mean_diff.item():.8f}")
            print(f"  Cosine similarity: {similarity.item():.8f}")


def main():
    config = BertConfig()
    num_classes = 2

    # Load base BERT model first, then create classification model
    bert_base = BertModel.from_pretrained('bert-base-uncased')
    pretrained_model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=num_classes)

    # Create custom model and load weights
    custom_model = CustomBertForSequenceClassification(
        config, num_classes=num_classes)
    custom_model.load_pretrained_weights(pretrained_model)
    print("Weights loaded successfully!")

    # Generate input
    input_ids, attention_mask, token_type_ids = generate_random_input()

    # Compare final outputs
    pretrained_model.eval()
    custom_model.eval()

    with torch.no_grad():
        pretrained_output = pretrained_model(
            input_ids, attention_mask, token_type_ids).logits
        custom_output, _ = custom_model(
            input_ids, attention_mask, token_type_ids)

        print("\nFinal output comparison:")
        print("Pretrained model output shape:", pretrained_output.shape)
        print("Custom model output shape:", custom_output.shape)
        difference = torch.abs(pretrained_output - custom_output).max()
        mean_diff = torch.abs(pretrained_output - custom_output).mean()
        similarity = F.cosine_similarity(
            pretrained_output.view(-1), custom_output.view(-1), dim=0)
        print(f"Maximum difference: {difference.item():.8f}")
        print(f"Mean difference: {mean_diff.item():.8f}")
        print(f"Output similarity: {similarity.item():.8f}")

    # Compare layer-by-layer
    compare_layer_outputs(pretrained_model, custom_model,
                          input_ids, attention_mask, token_type_ids)


if __name__ == "__main__":
    main()

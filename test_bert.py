import torch
from bert_dev import BertEmbeddings, BertConfig, BertAttention, BertEncoder
from transformers import BertModel
import numpy as np
import pytest


def set_seed(seed):
    """Ensure reproducibility by setting a fixed random seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# Load Hugging Face BERT for comparison
bert_base = BertModel.from_pretrained("bert-base-uncased")
bert_base.eval()


def generate_random_input():
    """Generate random token input for testing."""
    input_ids = torch.randint(0, 30522, (2, 10))  # batch_size=2, seq_len=10
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    return input_ids, attention_mask, token_type_ids


class TestBertEmbeddings:
    """🧪 Test class for verifying BERT embedding layers."""

    def setup_method(self):
        """⚙️ Setup before each test: Load custom BERT embeddings."""
        self.config = BertConfig()
        self.input_ids, _, self.token_type_ids = generate_random_input()

        # Initialize custom BERT embeddings
        self.emb = BertEmbeddings(self.config).load_from_pretrained()
        self.emb.eval()

    def test_word_embeddings(self):
        """📝 Test: Word embeddings match the pre-trained BERT model."""
        temp1 = bert_base.embeddings.word_embeddings(self.input_ids)
        temp2 = self.emb.word_embeddings(self.input_ids)

        assert torch.allclose(
            temp1, temp2, atol=1e-6), "❌ Word Embeddings Mismatch!"
        print("✅ Word Embeddings Match! 🎉")

    def test_position_embeddings(self):
        """📝 Test: Position embeddings match the pre-trained BERT model."""
        batch_size, seq_length = self.input_ids.size()

        # Generate position IDs
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=self.input_ids.device).unsqueeze(0).expand(batch_size, -1)

        temp1 = bert_base.embeddings.position_embeddings(position_ids)
        temp2 = self.emb.position_embeddings(position_ids)

        assert torch.allclose(
            temp1, temp2, atol=1e-6), "❌ Position Embeddings Mismatch!"
        print("✅ Position Embeddings Match! 🔥")

    def test_token_type_embeddings(self):
        """📝 Test: Token type embeddings match the pre-trained BERT model."""
        temp1 = bert_base.embeddings.token_type_embeddings(self.token_type_ids)
        temp2 = self.emb.token_type_embeddings(self.token_type_ids)

        assert torch.allclose(
            temp1, temp2, atol=1e-6), "❌ Token Type Embeddings Mismatch!"
        print("✅ Token Type Embeddings Match! 🚀")

    def test_combined_embeddings(self):
        """📝 Test: Final combined embeddings (word + position + token type) match the pre-trained model."""
        temp1 = bert_base.embeddings(
            input_ids=self.input_ids, token_type_ids=self.token_type_ids)
        temp2 = self.emb(self.input_ids, self.token_type_ids)

        assert torch.allclose(
            temp1, temp2, atol=1e-6), "❌ Combined Embeddings Mismatch!"
        print("✅ Combined Embeddings Match! 💯")


class TestBertAttention:
    """🧪 Test class for verifying BERT attention layers."""

    def setup_method(self):
        """⚙️ Setup before each test: Load custom BERT attention layer."""
        self.config = BertConfig()
        self.attn = BertAttention(self.config).load_from_pretrained()
        self.attn.eval()

        # Pretrained model
        self.hf_attention_0 = bert_base.encoder.layer[0].attention

        # Generate test input
        self.input_tensor = torch.rand(2, 128, 768)

    def test_attention_layer_0(self):
        """📝 Test: Self-attention output matches the pre-trained model."""
        temp1 = self.attn(self.input_tensor)
        temp2 = self.hf_attention_0(self.input_tensor)

        assert torch.allclose(
            temp1[0], temp2[0], atol=1e-5), "❌ Attention output Mismatch!"
        print("✅ Attention output Match! 🚀")


class TestBertEncoder:
    """🧪 Test class for verifying BERT encoder layers."""

    def setup_method(self):
        """⚙️ Setup before each test: Load custom BERT encoder."""
        self.config = BertConfig()

        # Initialize custom BERT encoder
        self.encoder = BertEncoder(self.config).load_from_pretrained()
        self.encoder.eval()

        # Load Hugging Face model encoder for reference
        self.hf_encoder = bert_base.encoder

        # Generate random input
        self.input_tensor = torch.rand(2, 128, 768)

    def test_all_attention_layers(self):
        """📝 Test: Each self-attention layer matches the pre-trained model."""
        for layer_idx in range(self.config.num_hidden_layers):
            custom_layer = self.encoder.layer[layer_idx].attention
            hf_layer = self.hf_encoder.layer[layer_idx].attention

            output_custom = custom_layer(self.input_tensor)
            output_hf = hf_layer(self.input_tensor)

            assert torch.allclose(
                output_custom[0], output_hf[0], atol=1e-5), f"❌ Attention Layer {layer_idx} Mismatch!"
            print(f"✅ Attention Layer {layer_idx} Match! 🎯")

    def test_all_encoder_layers(self):
        """📝 Test: Full BERT encoder output matches pre-trained model."""
        output_custom = self.encoder(self.input_tensor)
        output_hf = self.hf_encoder(self.input_tensor).last_hidden_state

        assert torch.allclose(
            output_custom, output_hf, atol=1e-5), "❌ Full Encoder Output Mismatch!"
        print("✅ Full Encoder Output Matches! 🎉")

    def test_intermediate_ffn_layers(self):
        """📝 Test: Intermediate feed-forward layers match the pre-trained model."""
        for layer_idx in range(self.config.num_hidden_layers):
            custom_ffn = self.encoder.layer[layer_idx].intermediate
            hf_ffn = self.hf_encoder.layer[layer_idx].intermediate

            output_custom = custom_ffn(self.input_tensor)
            output_hf = hf_ffn(self.input_tensor)

            assert torch.allclose(output_custom, output_hf,
                                  atol=1e-5), f"❌ FFN Layer {layer_idx} Mismatch!"
            print(f"✅ FFN Layer {layer_idx} Match! 🚀")


if __name__ == "__main__":
    print("\n🔍 Running BERT Embedding & Encoder Tests... 🚀\n")
    pytest.main(["-v", "--tb=short"])


import torch
from bert_dev import BertEmbeddings, BertConfig
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


class TestBertEncoder:
    """🧪 Test class for verifying BERT embedding layers."""

    def setup_method(self):
        """⚙️ Setup before each test: Load custom BERT embeddings."""
        self.config = BertConfig()
        self.input_ids, _, self.token_type_ids = generate_random_input()

        # Initialize custom BERT embeddings
        self.emb = BertEmbeddings(self.config).load_from_pretrained()
        self.emb.eval()

    def test_word_embeddings_1(self):
        """📝 Test: Word embeddings match the pre-trained BERT model."""
        self.we_fm = bert_base.embeddings.word_embeddings(self.input_ids)
        self.we_sm = self.emb.word_embeddings(self.input_ids)

        assert torch.allclose(
            self.we_fm, self.we_sm, atol=1e-6), "❌ Word Embeddings Mismatch!"
        print("✅ Word Embeddings Match! 🎉")

    def test_word_encoder1(self):
        """📝 Test: Word embeddings match the pre-trained BERT model."""
        encoder_hf = bert_base.encoder.layer[0]
        temp1 = encoder_hf(self.we_fm)

        # temp2 = self.emb.word_embeddings(self.input_ids)

        # assert torch.allclose(
        #     temp1, temp2, atol=1e-6), "❌ Word Embeddings Mismatch!"
        # print("✅ Word Embeddings Match! 🎉")


if __name__ == "__main__":
    print("\n🔍 Running BERT Embedding Tests... 🚀\n")
    pytest.main(["-v", "--tb=short"])

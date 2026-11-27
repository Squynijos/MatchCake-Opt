import pytest
from tokenizers import Tokenizer

from matchcake_opt.tokenizers import CHAT_SPECIAL_TOKENS, HFTokenizer


class TestLoading:
    def test_requires_name_or_tokenizer(self):
        with pytest.raises(ValueError):
            HFTokenizer()

    def test_from_file_and_directory(self, tmp_path, byte_level_hf_tokenizer):
        byte_level_hf_tokenizer.save(str(tmp_path / "tokenizer.json"))
        from_file = HFTokenizer(tmp_path / "tokenizer.json")
        from_dir = HFTokenizer(str(tmp_path))
        assert from_file.encode("hello world") == from_dir.encode("hello world")

    def test_from_hub(self, monkeypatch, byte_level_hf_tokenizer):
        calls = {}

        def fake_from_pretrained(identifier, revision):
            calls["args"] = (identifier, revision)
            return byte_level_hf_tokenizer

        monkeypatch.setattr(Tokenizer, "from_pretrained", staticmethod(fake_from_pretrained))
        tokenizer = HFTokenizer("someone/some-tokenizer", revision="v1")
        assert calls["args"] == ("someone/some-tokenizer", "v1")
        assert tokenizer.vocab_size > 0


class TestSpecialTokens:
    def test_existing_endoftext_used_as_bos(self, byte_level_hf_tokenizer):
        tokenizer = HFTokenizer(tokenizer=byte_level_hf_tokenizer)
        assert tokenizer.bos_token == "<|endoftext|>"
        assert set(CHAT_SPECIAL_TOKENS) <= set(tokenizer.special_tokens)
        assert tokenizer.vocab_size == 300 + len(CHAT_SPECIAL_TOKENS)
        assert max(tokenizer.special_tokens.values()) == tokenizer.vocab_size - 1

    def test_bos_added_when_missing(self, sentencepiece_like_hf_tokenizer):
        tokenizer = HFTokenizer(tokenizer=sentencepiece_like_hf_tokenizer)
        assert tokenizer.bos_token == "<|bos|>"
        assert tokenizer.bos_token_id == 7, "added after the original vocabulary"

    def test_custom_bos(self, byte_level_hf_tokenizer):
        tokenizer = HFTokenizer(tokenizer=byte_level_hf_tokenizer, bos_token="<|doc|>")
        assert tokenizer.bos_token_id == tokenizer.special_tokens["<|doc|>"]

    def test_special_text_is_not_parsed(self, byte_level_hf_tokenizer):
        tokenizer = HFTokenizer(tokenizer=byte_level_hf_tokenizer)
        ids = tokenizer.encode("<|user_start|> hi")
        assert tokenizer.encode_special("<|user_start|>") not in ids
        assert tokenizer.decode(ids) == "<|user_start|> hi"


class TestByteLevel:
    def test_encode_decode_round_trip(self, byte_level_hf_tokenizer):
        tokenizer = HFTokenizer(tokenizer=byte_level_hf_tokenizer)
        texts = ["hello fermions", "élégant, rapide, 速い."]
        assert [tokenizer.decode(ids) for ids in tokenizer.encode(texts)] == texts

    def test_decode_keeps_special_tokens(self, byte_level_hf_tokenizer):
        tokenizer = HFTokenizer(tokenizer=byte_level_hf_tokenizer)
        ids = tokenizer.encode("hello", prepend=tokenizer.bos_token)
        assert tokenizer.decode(ids) == "<|endoftext|>hello"

    def test_token_bytes_are_exact(self, byte_level_hf_tokenizer):
        tokenizer = HFTokenizer(tokenizer=byte_level_hf_tokenizer)
        token_bytes = tokenizer.token_bytes()
        assert all(token_bytes[i] == 0 for i in tokenizer.special_tokens.values())
        text = "The quick brown fox, élégant 速い!"
        assert int(token_bytes[tokenizer.encode(text)].sum()) == len(text.encode("utf-8"))


class TestNonByteLevel:
    def test_token_to_bytes(self, sentencepiece_like_hf_tokenizer):
        tokenizer = HFTokenizer(tokenizer=sentencepiece_like_hf_tokenizer)
        assert tokenizer.token_to_bytes(1) == b"\n", "byte-fallback token"
        assert tokenizer.token_to_bytes(2) == b" hello", "metaspace read as a space"
        assert tokenizer.token_to_bytes(4) == b"s", "wordpiece prefix removed"
        assert tokenizer.token_to_bytes(5) == "é".encode("utf-8")
        assert tokenizer.token_to_bytes(tokenizer.bos_token_id) == b""
        assert tokenizer.token_to_bytes(10_000) == b"", "unknown id"

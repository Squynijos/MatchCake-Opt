import pickle

import pytest
import tiktoken

from matchcake_opt.tokenizers import (
    CHAT_SPECIAL_TOKENS,
    NanochatTokenizer,
    TiktokenTokenizer,
)
from matchcake_opt.tokenizers.tiktoken_tokenizer import extend_encoding

from .conftest import make_tiktoken_encoding


class TestExtendEncoding:
    def test_adds_missing_tokens_after_vocab(self, tiny_encoding):
        extended = extend_encoding(tiny_encoding, ["<|endoftext|>", "<|a|>", "<|b|>", "<|a|>"])
        assert extended._special_tokens == {"<|endoftext|>": 259, "<|a|>": 260, "<|b|>": 261}
        assert extended.n_vocab == 262
        assert extended.encode_ordinary("hello") == tiny_encoding.encode_ordinary("hello")

    def test_returns_same_encoding_when_nothing_missing(self, tiny_encoding):
        assert extend_encoding(tiny_encoding, ["<|endoftext|>"]) is tiny_encoding


class TestTiktokenTokenizer:
    def test_defaults_to_endoftext_bos(self, tiny_encoding):
        tokenizer = TiktokenTokenizer(encoding=tiny_encoding)
        assert tokenizer.bos_token == "<|endoftext|>"
        assert tokenizer.bos_token_id == 259
        assert set(CHAT_SPECIAL_TOKENS) <= set(tokenizer.special_tokens)
        assert tokenizer.vocab_size == 260 + len(CHAT_SPECIAL_TOKENS)

    def test_adds_bos_when_encoding_has_none(self):
        tokenizer = TiktokenTokenizer(encoding=make_tiktoken_encoding(special_tokens={}))
        assert tokenizer.bos_token == "<|bos|>"
        assert tokenizer.bos_token_id == 259

    def test_custom_bos(self, tiny_encoding):
        tokenizer = TiktokenTokenizer(encoding=tiny_encoding, bos_token="<|start|>")
        assert tokenizer.bos_token_id == tokenizer.special_tokens["<|start|>"]

    def test_requires_name_or_encoding(self):
        with pytest.raises(ValueError):
            TiktokenTokenizer(name=None)

    def test_loads_named_encoding(self, monkeypatch, tiny_encoding):
        monkeypatch.setattr(tiktoken, "get_encoding", lambda name: tiny_encoding)
        assert TiktokenTokenizer("gpt2").vocab_size == 260 + len(CHAT_SPECIAL_TOKENS)

    def test_special_text_is_not_parsed(self, tiny_encoding):
        tokenizer = TiktokenTokenizer(encoding=tiny_encoding)
        ids = tokenizer.encode("<|user_start|>")
        assert tokenizer.encode_special("<|user_start|>") not in ids
        assert tokenizer.decode(ids) == "<|user_start|>"

    def test_encode_decode_batch(self, tiny_encoding):
        tokenizer = TiktokenTokenizer(encoding=tiny_encoding)
        texts = ["hello world", "Matchgates é 速"]
        batch = tokenizer.encode(texts, prepend=tokenizer.bos_token)
        assert [tokenizer.decode(ids[1:]) for ids in batch] == texts
        assert batch[0][1] == 258, "uses the merges of the encoding"

    def test_token_bytes(self, tiny_encoding):
        tokenizer = TiktokenTokenizer(encoding=tiny_encoding)
        token_bytes = tokenizer.token_bytes()
        assert token_bytes[ord("a")] == 1
        assert token_bytes[258] == 4
        assert all(token_bytes[i] == 0 for i in tokenizer.special_tokens.values())
        assert tokenizer.token_to_bytes(tokenizer.bos_token_id) == b""
        text = "hello world, élégant 速い"
        assert int(token_bytes[tokenizer.encode(text)].sum()) == len(text.encode("utf-8"))

    def test_unused_ids_count_zero_bytes(self):
        encoding = make_tiktoken_encoding(special_tokens={"<|endoftext|>": 262})  # ids 259-261 unused
        tokenizer = TiktokenTokenizer(encoding=encoding)
        assert tokenizer.token_to_bytes(260) == b""
        assert tokenizer.token_bytes()[260] == 0


class TestNanochatTokenizer:
    def write_pickle(self, directory, obj):
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "tokenizer.pkl", "wb") as f:
            pickle.dump(obj, f)

    def test_loads_directory_with_nanochat_ids(self, tmp_path):
        specials = {name: 259 + i for i, name in enumerate(["<|bos|>", *CHAT_SPECIAL_TOKENS])}
        self.write_pickle(tmp_path / "tok", make_tiktoken_encoding(special_tokens=specials))
        tokenizer = NanochatTokenizer(tmp_path / "tok")
        assert tokenizer.bos_token == "<|bos|>"
        assert tokenizer.special_tokens == specials, "ids must match the ones nanochat trained with"
        assert tokenizer.vocab_size == 259 + len(specials)
        assert tokenizer.tokenizer_dir == tmp_path / "tok"

    def test_missing_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            NanochatTokenizer(tmp_path / "missing")

    def test_wrong_pickle_content(self, tmp_path):
        self.write_pickle(tmp_path, {"not": "an encoding"})
        with pytest.raises(TypeError):
            NanochatTokenizer(tmp_path)

"""Small tokenizers built in memory so the tests never need the network."""

from typing import Dict, Optional

import pytest
import tiktoken
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

CORPUS = [
    "hello world, hello fermions!",
    "The quick brown fox jumps over the lazy dog.",
    "Matchgates are free-fermion circuits: élégant, rapide, 速い.",
] * 20


def make_tiktoken_encoding(special_tokens: Optional[Dict[str, int]] = None) -> tiktoken.Encoding:
    mergeable_ranks = {bytes([i]): i for i in range(256)}
    mergeable_ranks[b"he"] = 256
    mergeable_ranks[b"ll"] = 257
    mergeable_ranks[b"hell"] = 258
    if special_tokens is None:
        special_tokens = {"<|endoftext|>": 259}
    return tiktoken.Encoding(
        name="tiny", pat_str=GPT2_PATTERN, mergeable_ranks=mergeable_ranks, special_tokens=special_tokens
    )


@pytest.fixture
def tiny_encoding() -> tiktoken.Encoding:
    return make_tiktoken_encoding()


@pytest.fixture
def byte_level_hf_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(CORPUS, trainer=trainer)
    return tokenizer


@pytest.fixture
def sentencepiece_like_hf_tokenizer() -> Tokenizer:
    vocab = {"<unk>": 0, "<0x0A>": 1, "▁hello": 2, "▁world": 3, "##s": 4, "é": 5, "▁": 6}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()
    return tokenizer

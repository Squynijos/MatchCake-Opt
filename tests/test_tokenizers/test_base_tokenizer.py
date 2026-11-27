from typing import Dict, List, Sequence

import pytest
import torch

from matchcake_opt.tokenizers import (
    CHAT_SPECIAL_TOKENS,
    BaseTokenizer,
    TiktokenTokenizer,
)


class ByteTokenizer(BaseTokenizer):
    """One token per UTF-8 byte; special tokens after the 256 byte ids. Only implements the primitives."""

    SPECIALS = ["<|bos|>", *CHAT_SPECIAL_TOKENS]

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.SPECIALS)

    @property
    def bos_token(self) -> str:
        return "<|bos|>"

    @property
    def special_tokens(self) -> Dict[str, int]:
        return {token: 256 + i for i, token in enumerate(self.SPECIALS)}

    def encode_ordinary(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        inverse = {i: t for t, i in self.special_tokens.items()}
        out, buffer = [], bytearray()
        for i in ids:
            if i in inverse:
                out.append(buffer.decode("utf-8", errors="replace") + inverse[i])
                buffer = bytearray()
            else:
                buffer.append(i)
        return "".join(out) + buffer.decode("utf-8", errors="replace")

    def token_to_bytes(self, token_id: int) -> bytes:
        return bytes([token_id]) if token_id < 256 else b""


@pytest.fixture
def tokenizer() -> ByteTokenizer:
    return ByteTokenizer()


def ids_of(tokenizer: BaseTokenizer, *tokens: str) -> List[int]:
    return [tokenizer.encode_special(t) for t in tokens]


class TestEncode:
    def test_string(self, tokenizer):
        assert tokenizer.encode("hé") == list("hé".encode("utf-8"))

    def test_prepend_append_by_text_and_id(self, tokenizer):
        bos = tokenizer.bos_token_id
        end = tokenizer.encode_special("<|assistant_end|>")
        assert tokenizer.encode("a", prepend="<|bos|>", append=end) == [bos, ord("a"), end]

    def test_batch_uses_default_loop(self, tokenizer):
        bos = tokenizer.bos_token_id
        assert tokenizer.encode(["a", "bc"], prepend=bos) == [[bos, ord("a")], [bos, ord("b"), ord("c")]]

    def test_call_is_encode(self, tokenizer):
        assert tokenizer("ab") == tokenizer.encode("ab")

    def test_invalid_type(self, tokenizer):
        with pytest.raises(TypeError):
            tokenizer.encode(12)

    def test_unknown_special_token(self, tokenizer):
        with pytest.raises(KeyError):
            tokenizer.encode_special("<|nope|>")

    def test_len_and_repr(self, tokenizer):
        assert len(tokenizer) == tokenizer.vocab_size
        assert "ByteTokenizer" in repr(tokenizer) and "<|bos|>" in repr(tokenizer)


class TestTokenBytes:
    def test_values(self, tokenizer):
        token_bytes = tokenizer.token_bytes()
        assert token_bytes.dtype == torch.int32
        assert token_bytes.shape == (tokenizer.vocab_size,)
        assert torch.all(token_bytes[:256] == 1)
        assert torch.all(token_bytes[256:] == 0)

    def test_cached_and_device(self, tokenizer):
        assert tokenizer.token_bytes() is tokenizer.token_bytes()
        assert tokenizer.token_bytes(device=torch.device("cpu")).device.type == "cpu"


class TestRenderConversation:
    def test_simple_conversation_and_mask(self, tokenizer):
        conversation = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
            ]
        }
        ids, mask = tokenizer.render_conversation(conversation)
        us, ue, as_, ae = ids_of(
            tokenizer, "<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>"
        )
        bos = tokenizer.bos_token_id
        assert ids == [bos, us, ord("h"), ord("i"), ue, as_, ord("y"), ord("o"), ae]
        assert mask == [0, 0, 0, 0, 0, 0, 1, 1, 1]

    def test_system_message_merged(self, tokenizer):
        conversation = {
            "messages": [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "U"},
                {"role": "assistant", "content": "A"},
            ]
        }
        ids, _ = tokenizer.render_conversation(conversation)
        assert tokenizer.decode(ids).startswith("<|bos|><|user_start|>S\n\nU<|user_end|>")
        assert conversation["messages"][1]["content"] == "U", "the input must not be modified"

    def test_tool_parts(self, tokenizer):
        conversation = {
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "t"},
                        {"type": "python", "text": "p"},
                        {"type": "python_output", "text": "o"},
                    ],
                },
            ]
        }
        ids, mask = tokenizer.render_conversation(conversation)
        ps, pe, os_, oe, ae = ids_of(
            tokenizer, "<|python_start|>", "<|python_end|>", "<|output_start|>", "<|output_end|>", "<|assistant_end|>"
        )
        assert list(zip(ids, mask))[-8:] == [
            (ord("t"), 1),
            (ps, 1),
            (ord("p"), 1),
            (pe, 1),
            (os_, 0),
            (ord("o"), 0),
            (oe, 0),
            (ae, 1),
        ]

    def test_truncation(self, tokenizer):
        conversation = {"messages": [{"role": "user", "content": "x" * 50}]}
        ids, mask = tokenizer.render_conversation(conversation, max_tokens=10)
        assert len(ids) == len(mask) == 10

    @pytest.mark.parametrize(
        "messages, error",
        [
            ([], ValueError),
            ([{"role": "system", "content": "s"}], ValueError),
            ([{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}], ValueError),
            ([{"role": "assistant", "content": "a"}], ValueError),
            ([{"role": "user", "content": ["not", "a", "string"]}], TypeError),
            ([{"role": "user", "content": "u"}, {"role": "assistant", "content": 3}], TypeError),
            (
                [{"role": "user", "content": "u"}, {"role": "assistant", "content": [{"type": "?", "text": "a"}]}],
                ValueError,
            ),
        ],
    )
    def test_invalid_conversations(self, tokenizer, messages, error):
        with pytest.raises(error):
            tokenizer.render_conversation({"messages": messages})


class TestRenderForCompletion:
    def test_primes_assistant(self, tokenizer):
        conversation = {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "answer"},
            ]
        }
        ids = tokenizer.render_for_completion(conversation)
        assert tokenizer.decode(ids) == "<|bos|><|user_start|>q<|user_end|><|assistant_start|>"
        assert len(conversation["messages"]) == 2, "the input must not be modified"

    def test_requires_assistant_last(self, tokenizer):
        with pytest.raises(ValueError):
            tokenizer.render_for_completion({"messages": [{"role": "user", "content": "q"}]})


def test_round_trip_with_real_adapter(tiny_encoding):
    tokenizer = TiktokenTokenizer(encoding=tiny_encoding)
    conversation = {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hell"}]}
    ids, mask = tokenizer.render_conversation(conversation)
    assert tokenizer.decode(ids) == (
        "<|endoftext|><|user_start|>hello<|user_end|><|assistant_start|>hell<|assistant_end|>"
    )
    assert sum(mask) == len(tokenizer.encode("hell")) + 1

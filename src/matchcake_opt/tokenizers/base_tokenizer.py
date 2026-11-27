"""
Common interface for the tokenizers used by the language-model pipeline.

The chat rendering logic (``render_conversation`` and ``render_for_completion``) is adapted from
nanochat (https://github.com/karpathy/nanochat, MIT License, Copyright (c) 2025 Andrej Karpathy).
"""

import abc
import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, overload

import torch

BOS_TOKEN = "<|bos|>"
USER_START = "<|user_start|>"
USER_END = "<|user_end|>"
ASSISTANT_START = "<|assistant_start|>"
ASSISTANT_END = "<|assistant_end|>"
PYTHON_START = "<|python_start|>"
PYTHON_END = "<|python_end|>"
OUTPUT_START = "<|output_start|>"
OUTPUT_END = "<|output_end|>"

CHAT_SPECIAL_TOKENS: Tuple[str, ...] = (
    USER_START,
    USER_END,
    ASSISTANT_START,
    ASSISTANT_END,
    PYTHON_START,
    PYTHON_END,
    OUTPUT_START,
    OUTPUT_END,
)
"""Special tokens needed to render conversations, in the order they are added to a vocabulary that lacks them."""

Conversation = Dict[str, Any]
"""A conversation: ``{"messages": [{"role": "user" | "assistant" | "system", "content": ...}, ...]}``."""


class BaseTokenizer(abc.ABC):
    """
    Interface between a trained tokenizer and the language-model pipeline.

    Subclasses wrap an already trained tokenizer and implement the few primitive methods below.
    Everything the pipeline needs on top of that (BOS handling, batch encoding, the chat template with its
    loss mask, and the bytes-per-token table used by the bits-per-byte metric) is implemented here, so every
    tokenizer behaves the same way.

    Every tokenizer guarantees that the BOS token and all :data:`CHAT_SPECIAL_TOKENS` exist in its vocabulary;
    adapters add the missing ones after the trained vocabulary, which is why :attr:`vocab_size` can be larger
    than the vocabulary of the wrapped tokenizer.
    """

    TOKENIZER_NAME = "BaseTokenizer"

    def __init__(self) -> None:
        self._token_bytes: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ primitives to implement

    @property
    @abc.abstractmethod
    def vocab_size(self) -> int:
        """Number of token ids, special tokens included. Token ids are in ``[0, vocab_size)``."""

    @property
    @abc.abstractmethod
    def bos_token(self) -> str:
        """Text of the special token that starts every document."""

    @property
    @abc.abstractmethod
    def special_tokens(self) -> Dict[str, int]:
        """Mapping from every special token text to its id."""

    @abc.abstractmethod
    def encode_ordinary(self, text: str) -> List[int]:
        """Encode ``text`` without interpreting special tokens and without adding BOS."""

    @abc.abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        """Decode token ids to text. Special tokens are rendered as their text."""

    @abc.abstractmethod
    def token_to_bytes(self, token_id: int) -> bytes:
        """
        Raw bytes that ``token_id`` stands for in the original text.

        Return ``b""`` for special tokens and for ids that do not correspond to any token.
        """

    # ------------------------------------------------------------------ shared behaviour

    def encode_ordinary_batch(self, texts: Sequence[str]) -> List[List[int]]:
        """Encode several texts. Subclasses can override this with a faster batched implementation."""
        return [self.encode_ordinary(text) for text in texts]

    @property
    def bos_token_id(self) -> int:
        return self.encode_special(self.bos_token)

    def encode_special(self, token: str) -> int:
        """Id of the special token ``token``."""
        special_tokens = self.special_tokens
        if token not in special_tokens:
            raise KeyError(f"{token!r} is not a special token of {self.TOKENIZER_NAME}.")
        return special_tokens[token]

    @overload
    def encode(
        self, text: str, prepend: Optional[Union[int, str]] = None, append: Optional[Union[int, str]] = None
    ) -> List[int]: ...

    @overload
    def encode(
        self,
        text: List[str],
        prepend: Optional[Union[int, str]] = None,
        append: Optional[Union[int, str]] = None,
    ) -> List[List[int]]: ...

    def encode(
        self,
        text: Union[str, Sequence[str]],
        prepend: Optional[Union[int, str]] = None,
        append: Optional[Union[int, str]] = None,
    ) -> Union[List[int], List[List[int]]]:
        """
        Encode one text or a list of texts.

        :param text: A string, or a list of strings to encode as a batch.
        :param prepend: Optional token (id or special token text) added at the start of every encoded text,
            typically :attr:`bos_token`.
        :param append: Optional token (id or special token text) added at the end of every encoded text.
        :return: A list of ids for a string, a list of lists of ids for a sequence of strings.
        """
        prepend_ids = [] if prepend is None else [self._as_token_id(prepend)]
        append_ids = [] if append is None else [self._as_token_id(append)]
        if isinstance(text, str):
            return prepend_ids + self.encode_ordinary(text) + append_ids
        if isinstance(text, Sequence):
            return [prepend_ids + ids + append_ids for ids in self.encode_ordinary_batch(list(text))]
        raise TypeError(f"Expected a string or a sequence of strings, got {type(text)}.")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def __len__(self) -> int:
        return self.vocab_size

    def token_bytes(self, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
        """
        Number of bytes each token id stands for, as an ``int32`` tensor of shape ``(vocab_size,)``.

        Special tokens and unused ids count as 0 bytes, so they are excluded from the bits-per-byte metric.
        The table is computed once and cached.
        """
        if self._token_bytes is None:
            special_ids = set(self.special_tokens.values())
            self._token_bytes = torch.tensor(
                [0 if i in special_ids else len(self.token_to_bytes(i)) for i in range(self.vocab_size)],
                dtype=torch.int32,
            )
        return self._token_bytes.to(device)

    def render_conversation(self, conversation: Conversation, max_tokens: int = 2048) -> Tuple[List[int], List[int]]:
        """
        Tokenize a conversation with the chat template.

        Template: ``<bos> <user_start> ... <user_end> <assistant_start> ... <assistant_end> ...``.
        A leading system message is merged into the first user message. Assistant content is either a string or
        a list of parts ``{"type": "text" | "python" | "python_output", "text": ...}``; python tool calls are
        wrapped in ``<python_start>``/``<python_end>`` and tool outputs in ``<output_start>``/``<output_end>``.

        :param conversation: ``{"messages": [...]}`` with alternating user / assistant messages.
        :param max_tokens: Both outputs are truncated to this length.
        :return: ``(ids, mask)`` where ``mask[i] == 1`` for the tokens the assistant is trained to produce
            (assistant text, tool calls and ``<assistant_end>``), and 0 otherwise (prompt and tool outputs).
        """
        ids: List[int] = []
        mask: List[int] = []

        def add_tokens(token_ids: Union[int, List[int]], mask_val: int) -> None:
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        messages = conversation["messages"]
        if len(messages) > 0 and messages[0]["role"] == "system":
            messages = copy.deepcopy(messages)
            if len(messages) < 2 or messages[1]["role"] != "user":
                raise ValueError("A system message must be followed by a user message.")
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        if len(messages) < 1:
            raise ValueError("A conversation must contain at least one message.")

        add_tokens(self.bos_token_id, 0)
        for i, message in enumerate(messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            if message["role"] != expected_role:
                raise ValueError(f"Message {i} is from {message['role']!r} but should be from {expected_role!r}.")
            content = message["content"]
            if message["role"] == "user":
                if not isinstance(content, str):
                    raise TypeError("User messages must be strings.")
                add_tokens(self.encode_special(USER_START), 0)
                add_tokens(self.encode_ordinary(content), 0)
                add_tokens(self.encode_special(USER_END), 0)
                continue

            add_tokens(self.encode_special(ASSISTANT_START), 0)
            if isinstance(content, str):
                add_tokens(self.encode_ordinary(content), 1)
            elif isinstance(content, list):
                for part in content:
                    part_ids = self.encode_ordinary(part["text"])
                    if part["type"] == "text":
                        add_tokens(part_ids, 1)
                    elif part["type"] == "python":
                        add_tokens(self.encode_special(PYTHON_START), 1)
                        add_tokens(part_ids, 1)
                        add_tokens(self.encode_special(PYTHON_END), 1)
                    elif part["type"] == "python_output":
                        # tool outputs come from Python at inference time: never supervised
                        add_tokens(self.encode_special(OUTPUT_START), 0)
                        add_tokens(part_ids, 0)
                        add_tokens(self.encode_special(OUTPUT_END), 0)
                    else:
                        raise ValueError(f"Unknown assistant content part type: {part['type']!r}.")
            else:
                raise TypeError(f"Unknown assistant content type: {type(content)}.")
            add_tokens(self.encode_special(ASSISTANT_END), 1)

        return ids[:max_tokens], mask[:max_tokens]

    def render_for_completion(self, conversation: Conversation, max_tokens: int = 2048) -> List[int]:
        """
        Tokenize a conversation whose last message is the assistant's, without that message, and end with
        ``<assistant_start>`` so the model is primed to write the answer. Used by generative evaluation and RL.
        """
        messages = conversation["messages"]
        if len(messages) == 0 or messages[-1]["role"] != "assistant":
            raise ValueError("The last message must be from the assistant.")
        conversation = {**conversation, "messages": messages[:-1]}
        ids, _ = self.render_conversation(conversation, max_tokens=max_tokens)
        return ids + [self.encode_special(ASSISTANT_START)]

    def _as_token_id(self, token: Union[int, str]) -> int:
        return token if isinstance(token, int) else self.encode_special(token)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vocab_size={self.vocab_size}, bos_token={self.bos_token!r})"

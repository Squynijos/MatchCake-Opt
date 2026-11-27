import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import tiktoken

from .base_tokenizer import BOS_TOKEN, CHAT_SPECIAL_TOKENS, BaseTokenizer


def extend_encoding(encoding: tiktoken.Encoding, new_special_tokens: Sequence[str]) -> tiktoken.Encoding:
    """
    Return a copy of ``encoding`` with the missing ``new_special_tokens`` added after its last token id.

    Tokens already present are kept with their id. This follows the "extending tiktoken" recipe of the
    tiktoken README.
    """
    special_tokens = dict(encoding._special_tokens)
    missing = [token for token in dict.fromkeys(new_special_tokens) if token not in special_tokens]
    if not missing:
        return encoding
    next_id = encoding.max_token_value + 1
    for offset, token in enumerate(missing):
        special_tokens[token] = next_id + offset
    return tiktoken.Encoding(
        name=f"{encoding.name}_chat",
        pat_str=encoding._pat_str,
        mergeable_ranks=encoding._mergeable_ranks,
        special_tokens=special_tokens,
    )


class TiktokenTokenizer(BaseTokenizer):
    """
    Tokenizer backed by a `tiktoken <https://github.com/openai/tiktoken>`_ encoding.

    Use a named OpenAI encoding (``"gpt2"``, ``"r50k_base"``, ``"cl100k_base"``, ``"o200k_base"``, ...) or pass a
    ``tiktoken.Encoding`` directly. The chat special tokens and the BOS token are added after the original
    vocabulary when missing.

    Example config::

        tokenizer:
          class_path: matchcake_opt.tokenizers.TiktokenTokenizer
          init_args: {name: gpt2}
    """

    TOKENIZER_NAME = "TiktokenTokenizer"
    DEFAULT_BOS_TOKEN = "<|endoftext|>"
    """tiktoken's OpenAI encodings use ``<|endoftext|>`` as their document delimiter."""

    def __init__(
        self,
        name: Optional[str] = "gpt2",
        *,
        encoding: Optional[tiktoken.Encoding] = None,
        bos_token: Optional[str] = None,
        num_threads: int = 8,
    ):
        """
        :param name: Name of a tiktoken encoding, ignored when ``encoding`` is given.
        :param encoding: A ready ``tiktoken.Encoding``.
        :param bos_token: Special token that starts every document. Defaults to ``<|endoftext|>`` if the encoding
            has it, else ``<|bos|>`` (added if missing).
        :param num_threads: Threads used to encode batches of texts.
        """
        super().__init__()
        if encoding is None:
            if name is None:
                raise ValueError("Either `name` or `encoding` must be given.")
            encoding = tiktoken.get_encoding(name)
        if bos_token is None:
            bos_token = self.DEFAULT_BOS_TOKEN if self.DEFAULT_BOS_TOKEN in encoding._special_tokens else BOS_TOKEN
        self._bos_token = bos_token
        self._num_threads = num_threads
        self.encoding = extend_encoding(encoding, [bos_token, *CHAT_SPECIAL_TOKENS])
        self._special_ids = frozenset(self.encoding._special_tokens.values())

    @property
    def vocab_size(self) -> int:
        return self.encoding.n_vocab

    @property
    def bos_token(self) -> str:
        return self._bos_token

    @property
    def special_tokens(self) -> Dict[str, int]:
        return dict(self.encoding._special_tokens)

    def encode_ordinary(self, text: str) -> List[int]:
        return self.encoding.encode_ordinary(text)

    def encode_ordinary_batch(self, texts: Sequence[str]) -> List[List[int]]:
        return self.encoding.encode_ordinary_batch(list(texts), num_threads=self._num_threads)

    def decode(self, ids: Sequence[int]) -> str:
        return self.encoding.decode(list(ids))

    def token_to_bytes(self, token_id: int) -> bytes:
        if token_id in self._special_ids:
            return b""
        try:
            return self.encoding.decode_single_token_bytes(token_id)
        except KeyError:  # some encodings have unused ids (e.g. cl100k_base between its special tokens)
            return b""


class NanochatTokenizer(TiktokenTokenizer):
    """
    Tokenizer trained with nanochat's ``scripts/tok_train.py``.

    nanochat saves its tokenizer as a pickled ``tiktoken.Encoding`` in ``<tokenizer_dir>/tokenizer.pkl``
    (by default ``~/.cache/nanochat/tokenizer``). It already contains ``<|bos|>`` and the chat special tokens, so
    token ids match the ones used by nanochat.

    .. warning:: Loading a pickle can run arbitrary code: only load tokenizer directories you trust.

    Example config::

        tokenizer:
          class_path: matchcake_opt.tokenizers.NanochatTokenizer
          init_args: {tokenizer_dir: ~/.cache/nanochat/tokenizer}
    """

    TOKENIZER_NAME = "NanochatTokenizer"
    PICKLE_FILENAME = "tokenizer.pkl"

    def __init__(self, tokenizer_dir: Union[str, Path], *, num_threads: int = 8):
        """
        :param tokenizer_dir: Directory containing ``tokenizer.pkl``.
        :param num_threads: Threads used to encode batches of texts.
        """
        pickle_path = Path(tokenizer_dir).expanduser() / self.PICKLE_FILENAME
        if not pickle_path.exists():
            raise FileNotFoundError(f"No nanochat tokenizer found at {pickle_path}.")
        with open(pickle_path, "rb") as f:
            encoding = pickle.load(f)
        if not isinstance(encoding, tiktoken.Encoding):
            raise TypeError(f"{pickle_path} does not contain a tiktoken.Encoding (got {type(encoding)}).")
        self.tokenizer_dir = pickle_path.parent
        super().__init__(name=None, encoding=encoding, bos_token=BOS_TOKEN, num_threads=num_threads)

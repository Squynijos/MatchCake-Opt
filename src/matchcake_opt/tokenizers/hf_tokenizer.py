import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from tokenizers import AddedToken, Tokenizer, decoders

from .base_tokenizer import BOS_TOKEN, CHAT_SPECIAL_TOKENS, BaseTokenizer

_BYTE_FALLBACK_TOKEN = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


@lru_cache(maxsize=1)
def _byte_level_alphabet_to_byte() -> Dict[str, int]:
    """Inverse of GPT-2's ``bytes_to_unicode`` table used by byte-level BPE tokenizers."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    printable += list(range(ord("®"), ord("ÿ") + 1))
    byte_values = printable[:]
    chars = printable[:]
    n = 0
    for b in range(256):
        if b not in printable:
            byte_values.append(b)
            chars.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(byte_values, chars)}


class HFTokenizer(BaseTokenizer):
    """
    Tokenizer backed by a Hugging Face `tokenizers <https://github.com/huggingface/tokenizers>`_ tokenizer.

    Loads a ``tokenizer.json`` from a local file or directory, or from the Hugging Face Hub by repository id.
    The chat special tokens and the BOS token are added after the original vocabulary when missing.

    Example config::

        tokenizer:
          class_path: matchcake_opt.tokenizers.HFTokenizer
          init_args: {name_or_path: openai-community/gpt2}
    """

    TOKENIZER_NAME = "HFTokenizer"
    BOS_CANDIDATES = (BOS_TOKEN, "<|endoftext|>", "<|begin_of_text|>", "<s>", "<bos>")
    """Special tokens recognised as the document delimiter, in order of preference, when ``bos_token`` is not set."""

    def __init__(
        self,
        name_or_path: Optional[Union[str, Path]] = None,
        *,
        tokenizer: Optional[Tokenizer] = None,
        revision: str = "main",
        bos_token: Optional[str] = None,
    ):
        """
        :param name_or_path: Path to a ``tokenizer.json`` file, to a directory containing one, or a Hub repository
            id. Ignored when ``tokenizer`` is given.
        :param tokenizer: A ready ``tokenizers.Tokenizer``. It is modified in place (special tokens are added).
        :param revision: Hub revision, when loading from the Hub.
        :param bos_token: Special token that starts every document. Defaults to the first of
            :attr:`BOS_CANDIDATES` that is a special token of the tokenizer, else ``<|bos|>`` (added).
        """
        super().__init__()
        if tokenizer is None:
            if name_or_path is None:
                raise ValueError("Either `name_or_path` or `tokenizer` must be given.")
            tokenizer = self._load(name_or_path, revision)
        self.tokenizer = tokenizer
        # text passed to `encode_ordinary` must never be parsed as special tokens
        self.tokenizer.encode_special_tokens = True

        if bos_token is None:
            existing = self._read_special_tokens()
            bos_token = next((token for token in self.BOS_CANDIDATES if token in existing), BOS_TOKEN)
        self._bos_token = bos_token
        existing = self._read_special_tokens()
        missing = [t for t in dict.fromkeys([bos_token, *CHAT_SPECIAL_TOKENS]) if t not in existing]
        if missing:
            self.tokenizer.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in missing])

        self._special_tokens = self._read_special_tokens()
        self._vocab_size = max(self.tokenizer.get_vocab(with_added_tokens=True).values()) + 1
        self._byte_level = isinstance(self.tokenizer.decoder, decoders.ByteLevel)

    @staticmethod
    def _load(name_or_path: Union[str, Path], revision: str) -> Tokenizer:
        path = Path(name_or_path).expanduser()
        if path.is_dir():
            path = path / "tokenizer.json"
        if path.is_file():
            return Tokenizer.from_file(str(path))
        return Tokenizer.from_pretrained(str(name_or_path), revision=revision)

    def _read_special_tokens(self) -> Dict[str, int]:
        return {
            added.content: token_id
            for token_id, added in self.tokenizer.get_added_tokens_decoder().items()
            if added.special
        }

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def bos_token(self) -> str:
        return self._bos_token

    @property
    def special_tokens(self) -> Dict[str, int]:
        return dict(self._special_tokens)

    def encode_ordinary(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def encode_ordinary_batch(self, texts: Sequence[str]) -> List[List[int]]:
        return [encoding.ids for encoding in self.tokenizer.encode_batch(list(texts), add_special_tokens=False)]

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def token_to_bytes(self, token_id: int) -> bytes:
        """
        Bytes of a token. Exact for byte-level BPE tokenizers (GPT-2, Llama 3, Qwen, ...) and for SentencePiece
        byte-fallback tokens; for other tokenizers it is the UTF-8 encoding of the token text, with ``▁`` read as a
        space and a WordPiece ``##`` prefix removed.
        """
        token = self.tokenizer.id_to_token(token_id)
        if token is None or token in self._special_tokens:
            return b""
        if self._byte_level:
            alphabet = _byte_level_alphabet_to_byte()
            if all(char in alphabet for char in token):
                return bytes(alphabet[char] for char in token)
        match = _BYTE_FALLBACK_TOKEN.match(token)
        if match:
            return bytes([int(match.group(1), 16)])
        if token.startswith("##"):
            token = token[2:]
        return token.replace("▁", " ").encode("utf-8")

import abc
import tokenizers
import typing as t

from .base import BaseTokenizer


class HuggingTokenizer(BaseTokenizer, metaclass=abc.ABCMeta):
    """
    Abstract tokenizer to implement any tokenizer based on HuggingFace `tokenizers.Tokenizer`.
    """

    def __init__(self):
        super().__init__()
        self.hug: tokenizers.Tokenizer = self._build_hugging_tokenizer()

    def _build_hugging_tokenizer(self) -> tokenizers.Tokenizer:
        raise NotImplementedError()

    def tokenize(self, text: str) -> t.Iterator[str]:
        return self.hug.encode(text).tokens


class HuggingBertTokenizer(HuggingTokenizer):
    """
    Tokenizer based on the `bert-base-cased <https://huggingface.co/bert-base-cased>`_ tokenizer.
    """

    def _build_hugging_tokenizer(self):
        return tokenizers.Tokenizer.from_pretrained("bert-base-cased")


__all__ = (
    "HuggingBertTokenizer",
)

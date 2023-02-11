import abc
import tokenizers

from .base import BaseTokenizer


class HuggingTokenizer(BaseTokenizer, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
        self.hug: tokenizers.Tokenizer = self._build_hugging_tokenizer()

    def _build_hugging_tokenizer(self) -> tokenizers.Tokenizer:
        raise NotImplementedError()

    def tokenize_plain(self, text: str) -> str:
        return " ".join(self.hug.encode(text).tokens)


class HuggingBertTokenizer(HuggingTokenizer):
    def _build_hugging_tokenizer(self):
        return tokenizers.Tokenizer.from_pretrained("bert-base-cased")


__all__ = (
    "HuggingBertTokenizer",
)

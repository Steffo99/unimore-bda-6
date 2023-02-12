import typing as t

from .base import BaseTokenizer


class LowercaseTokenizer(BaseTokenizer):
    """
    Tokenizer which converts the words to lowercase before splitting them with `str.split`.
    """

    def tokenize(self, text: str) -> t.Iterator[str]:
        text = text.lower()
        tokens = text.split()
        return tokens

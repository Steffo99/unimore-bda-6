import typing as t

from .base import BaseTokenizer


class PlainTokenizer(BaseTokenizer):
    """
    Tokenizer which just splits the text into tokens by separating them at whitespaces with `str.split`.
    """

    def tokenize(self, text: str) -> t.Iterator[str]:
        tokens = text.split()
        return tokens

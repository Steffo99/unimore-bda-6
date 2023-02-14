import typing as t
import nltk
import nltk.sentiment.util

from .base import BaseTokenizer


class NLTKWordTokenizer(BaseTokenizer):
    """
    Tokenizer based on `nltk.word_tokenize`.
    """

    def tokenize(self, text: str) -> t.Iterator[str]:
        tokens = nltk.word_tokenize(text)
        nltk.sentiment.util.mark_negation(tokens, shallow=True)
        return tokens


__all__ = (
    "NLTKWordTokenizer",
)

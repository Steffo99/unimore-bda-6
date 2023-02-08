import nltk
import nltk.sentiment.util
import typing as t

from .base import BaseTokenizer


class NLTKWordTokenizer(BaseTokenizer):
    """
    Tokenizer based on `nltk.word_tokenize`.
    """

    def tokenize_plain(self, text: str) -> t.Iterable[str]:
        tokens = nltk.word_tokenize(text)
        nltk.sentiment.util.mark_negation(tokens, shallow=True)
        return tokens


__all__ = (
    "NLTKWordTokenizer",
)

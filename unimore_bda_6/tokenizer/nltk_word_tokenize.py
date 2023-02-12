import nltk
import nltk.sentiment.util

from .base import BaseTokenizer


class NLTKWordTokenizer(BaseTokenizer):
    """
    Tokenizer based on `nltk.word_tokenize`.
    """

    def tokenize(self, text: str) -> str:
        tokens = nltk.word_tokenize(text)
        nltk.sentiment.util.mark_negation(tokens, shallow=True)
        return " ".join(tokens)


__all__ = (
    "NLTKWordTokenizer",
)

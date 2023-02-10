import tensorflow

from .base import BaseTokenizer


class PlainTokenizer(BaseTokenizer):
    """
    Tokenizer which just splits the text into tokens by separating them at whitespaces.
    """

    def tokenize_plain(self, text: str) -> str:
        return text

    def tokenize_tensorflow(self, text: tensorflow.Tensor) -> tensorflow.Tensor:
        return text

import tensorflow

from .base import BaseTokenizer


class PlainTokenizer(BaseTokenizer):
    """
    Tokenizer which just splits the text into tokens by separating them at whitespaces.
    """

    def tokenize_plain(self, text: str) -> list[str]:
        return text.split()

    def tokenize_tensorflow(self, text: tensorflow.Tensor) -> tensorflow.Tensor:
        text = tensorflow.expand_dims(text, -1, name="tokens")
        return text

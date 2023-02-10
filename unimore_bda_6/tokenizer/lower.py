import tensorflow

from .base import BaseTokenizer


class LowercaseTokenizer(BaseTokenizer):
    """
    Tokenizer which converts the words to lowercase before splitting them via spaces.
    """

    def tokenize_plain(self, text: str) -> str:
        text = text.lower()
        return text

    def tokenize_tensorflow(self, text: tensorflow.Tensor) -> tensorflow.Tensor:
        text = tensorflow.strings.lower(text)
        return text

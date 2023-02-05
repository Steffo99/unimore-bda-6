import tensorflow

from .base import BaseTokenizer


class LowercaseTokenizer(BaseTokenizer):
    def tokenize_builtins(self, text: str) -> list[str]:
        return text.lower().split()

    def tokenize_tensorflow(self, text: tensorflow.Tensor) -> tensorflow.Tensor:
        text = tensorflow.strings.lower(text)
        text = tensorflow.expand_dims(text, -1, name="tokens")
        return text

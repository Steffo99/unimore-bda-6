import tensorflow

from .base import BaseTokenizer


class LowercaseTokenizer(BaseTokenizer):
    def tokenize_builtins(self, text: str) -> list[str]:
        return text.lower().split()

    def tokenize_tensorflow(self, text: tensorflow.Tensor) -> tensorflow.Tensor:
        return tensorflow.strings.lower(text)

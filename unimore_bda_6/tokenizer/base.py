import tensorflow


class BaseTokenizer:
    """
    The base for all tokenizers in this project.
    """

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"

    @staticmethod
    def __not_implemented(f):
        f.__notimplemented__ = True
        return f

    def can_tokenize_builtins(self) -> bool:
        return getattr(self.tokenize_builtins, "__notimplemented__", False)

    def can_tokenize_tensorflow(self) -> bool:
        return getattr(self.tokenize_tensorflow, "__notimplemented__", False)

    @__not_implemented
    def tokenize_builtins(self, text: str) -> list[str]:
        """
        Convert a text string into a list of tokens.
        """
        raise NotImplementedError()

    @__not_implemented
    def tokenize_tensorflow(self, text: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        Convert a `tensorflow.Tensor` string into another `tensorflow.Tensor` space-separated string.
        """
        raise NotImplementedError()

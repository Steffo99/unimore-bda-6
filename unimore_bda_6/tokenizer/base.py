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

    def supports_plain(self) -> bool:
        return not getattr(self.tokenize_plain, "__notimplemented__", False)

    def supports_tensorflow(self) -> bool:
        return not getattr(self.tokenize_tensorflow, "__notimplemented__", False)

    @__not_implemented
    def tokenize_plain(self, text: str) -> list[str]:
        """
        Convert a text string into a list of tokens.
        """
        raise NotImplementedError()

    @__not_implemented
    def tokenize_tensorflow(self, text: "tensorflow.Tensor") -> "tensorflow.Tensor":
        """
        Convert a `tensorflow.Tensor` string into another `tensorflow.Tensor` space-separated string.
        """
        raise NotImplementedError()

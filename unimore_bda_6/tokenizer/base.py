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
    def tokenize_plain(self, text: str) -> str:
        """
        Convert a text `str` into another `str` containing a series of whitespace-separated tokens.
        """
        raise NotImplementedError()

    def tokenize_and_split_plain(self, text: str) -> list[str]:
        """
        Run `.tokenize_plain`, then split the result using `str.split`.
        """
        text = self.tokenize_plain(text)
        text = text.split()
        return text

    @__not_implemented
    def tokenize_tensorflow(self, text: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        Convert a `tensorflow.Tensor` string into another `tensorflow.Tensor` space-separated string.
        """
        raise NotImplementedError()

    def tokenize_tensorflow_and_expand_dims(self, text: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        Run `.tokenize_tensorflow`, then add a dimension to the tensor for reasons unknown to me, but required to get `tensorflow` to work properly.
        """
        text = self.tokenize_tensorflow(text)
        text = tensorflow.expand_dims(text, -1, name="tokens")
        return text

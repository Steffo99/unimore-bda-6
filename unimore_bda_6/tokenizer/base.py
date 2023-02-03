import abc


class BaseTokenizer(metaclass=abc.ABCMeta):
    """
    The base for all tokenizers in this project.
    """

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"

    @abc.abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """
        Convert a text string into a list of tokens.
        """
        raise NotImplementedError()

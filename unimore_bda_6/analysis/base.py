import abc
import typing as t


Input = t.TypeVar("Input")
Category = t.TypeVar("Category")


class BaseSA(metaclass=abc.ABCMeta):
    """
    Abstract base class for sentiment analyzers implemented in this project.
    """

    @abc.abstractmethod
    def train(self, training_set: list[tuple[Input, Category]]) -> None:
        """
        Train the analyzer with the given training set.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def use(self, text: Input) -> Category:
        """
        Use the sentiment analyzer.
        """
        raise NotImplementedError()


class AlreadyTrainedError(Exception):
    """
    This model has already been trained and cannot be trained again.
    """

class NotTrainedError(Exception):
    """
    This model has not been trained yet.
    """


__all__ = (
    "Input",
    "Category",
    "BaseSA",
    "AlreadyTrainedError",
    "NotTrainedError",
)

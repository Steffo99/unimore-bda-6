import abc


class BaseSA(metaclass=abc.ABCMeta):
    """
    Abstract base class for sentiment analyzers implemented in this project.
    """

    def __init__(self) -> None:
        """
        Create the empty shell of the sentiment analyzer.
        """

        self.trained = False
        "If :meth:`train` has been called at least once, and the analyzer is ready or not to be evaluated or used."

    @abc.abstractmethod
    def train(self, training_set) -> None:
        """
        Train the analyzer with the given training set.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def evaluate(self, test_set) -> None:
        """
        Evaluate the analyzer with the given test set.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def use(self, text: str) -> str:
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
    "BaseSA",
    "AlreadyTrainedError",
    "NotTrainedError",
)

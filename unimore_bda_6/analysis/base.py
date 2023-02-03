import abc
import logging

from ..database import DataSet, Text, Category

log = logging.getLogger(__name__)


class BaseSentimentAnalyzer(metaclass=abc.ABCMeta):
    """
    Abstract base class for sentiment analyzers implemented in this project.
    """

    @abc.abstractmethod
    def train(self, training_set: DataSet) -> None:
        """
        Train the analyzer with the given training dataset.
        """
        raise NotImplementedError()

    def evaluate(self, test_set: DataSet) -> tuple[int, int]:
        """
        Perform a model evaluation by calling repeatedly `.use` on every text of the test dataset and by comparing its resulting category with the expected category.

        Returns a tuple with the number of correct results and the number of evaluated results.
        """
        evaluated: int = 0
        correct: int   = 0

        for text, expected_category in test_set:
            resulting_category = self.use(text)
            evaluated += 1
            correct += 1 if resulting_category == expected_category else 0
            if not evaluated % 100:
                log.debug("%d evaluated, %d correct, %0.2d %% accuracy", evaluated, correct, correct / evaluated * 100)

        return correct, evaluated

    @abc.abstractmethod
    def use(self, text: Text) -> Category:
        """
        Run the model on the given input.
        """
        raise NotImplementedError()


__all__ = (
    "BaseSentimentAnalyzer",
)

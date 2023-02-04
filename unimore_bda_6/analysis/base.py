import abc
import logging
import typing as t
import dataclasses

from ..database import Text, Category, Review, DatasetFunc

log = logging.getLogger(__name__)


@dataclasses.dataclass
class EvaluationResults:
    correct: int
    evaluated: int

    def __repr__(self):
        return f"<EvaluationResults: {self.correct}/{self.evaluated}, {self.correct / self.evaluated * 100:.2f}>"

    def __str__(self):
        return f"{self.correct} / {self.evaluated} - {self.correct / self.evaluated * 100:.2f} %"


class BaseSentimentAnalyzer(metaclass=abc.ABCMeta):
    """
    Abstract base class for sentiment analyzers implemented in this project.
    """

    @abc.abstractmethod
    def train(self, dataset_func: DatasetFunc) -> None:
        """
        Train the analyzer with the given training dataset.
        """
        raise NotImplementedError()

    def evaluate(self, dataset_func: DatasetFunc) -> EvaluationResults:
        """
        Perform a model evaluation by calling repeatedly `.use` on every text of the test dataset and by comparing its resulting category with the expected category.

        Returns a tuple with the number of correct results and the number of evaluated results.
        """

        evaluated: int = 0
        correct: int = 0

        for review in dataset_func():
            resulting_category = self.use(review.text)
            evaluated += 1
            correct += 1 if resulting_category == review.category else 0
            if not evaluated % 100:
                log.debug("%d evaluated, %d correct, %0.2d %% accuracy", evaluated, correct, correct / evaluated * 100)

        return EvaluationResults(correct=correct, evaluated=evaluated)

    @abc.abstractmethod
    def use(self, text: Text) -> Category:
        """
        Run the model on the given input.
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
    "BaseSentimentAnalyzer",
    "AlreadyTrainedError",
    "NotTrainedError",
)

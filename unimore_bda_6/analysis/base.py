from __future__ import annotations

import abc
import logging
import dataclasses

from ..database import Text, Category, CachedDatasetFunc
from ..tokenizer import BaseTokenizer

log = logging.getLogger(__name__)


class BaseSentimentAnalyzer(metaclass=abc.ABCMeta):
    """
    Abstract base class for sentiment analyzers implemented in this project.
    """

    # noinspection PyUnusedLocal
    def __init__(self, *, tokenizer: BaseTokenizer):
        pass

    def __repr__(self):
        return f"<{self.__class__.__qualname__}>"

    @abc.abstractmethod
    def train(self, training_dataset_func: CachedDatasetFunc, validation_dataset_func: CachedDatasetFunc) -> None:
        """
        Train the analyzer with the given training and validation datasets.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def use(self, text: Text) -> Category:
        """
        Run the model on the given input.
        """
        raise NotImplementedError()

    def evaluate(self, evaluation_dataset_func: CachedDatasetFunc) -> EvaluationResults:
        """
        Perform a model evaluation by calling repeatedly `.use` on every text of the test dataset and by comparing its resulting category with the expected category.

        Returns a tuple with the number of correct results and the number of evaluated results.
        """

        evaluated: int = 0
        correct: int = 0
        score: float = 0.0

        for review in evaluation_dataset_func():
            resulting_category = self.use(review.text)
            evaluated += 1
            correct += 1 if resulting_category == review.category else 0
            score += 1 - (abs(resulting_category - review.category) / 4)

        return EvaluationResults(correct=correct, evaluated=evaluated, score=score)


@dataclasses.dataclass
class EvaluationResults:
    """
    Container for the results of a dataset evaluation.
    """

    correct: int
    evaluated: int
    score: float

    def __repr__(self):
        return f"<EvaluationResults: {self!s}>"

    def __str__(self):
        return f"{self.evaluated} evaluated, {self.correct} correct, {self.correct / self.evaluated:.2%} accuracy, {self.score:.2f} score, {self.score / self.evaluated:.2%} scoreaccuracy"


class AlreadyTrainedError(Exception):
    """
    This model has already been trained and cannot be trained again.
    """


class NotTrainedError(Exception):
    """
    This model has not been trained yet.
    """


class TrainingFailedError(Exception):
    """
    The model wasn't able to complete the training and should not be used anymore.
    """


__all__ = (
    "BaseSentimentAnalyzer",
    "AlreadyTrainedError",
    "NotTrainedError",
    "TrainingFailedError",
)

from __future__ import annotations

import abc
import logging
import dataclasses

from ..database import CachedDatasetFunc
from ..tokenizer import BaseTokenizer

log = logging.getLogger(__name__)


class BaseSentimentAnalyzer(metaclass=abc.ABCMeta):
    """
    Abstract base class for sentiment analyzers implemented in this project.
    """

    def __init__(self, *, tokenizer: BaseTokenizer):
        self.tokenizer: BaseTokenizer = tokenizer

    def __repr__(self):
        return f"<{self.__class__.__qualname__} with {self.tokenizer} tokenizer>"

    @abc.abstractmethod
    def train(self, training_dataset_func: CachedDatasetFunc, validation_dataset_func: CachedDatasetFunc) -> None:
        """
        Train the analyzer with the given training and validation datasets.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def use(self, text: str) -> float:
        """
        Run the model on the given input, and return the predicted rating.
        """
        raise NotImplementedError()

    def evaluate(self, evaluation_dataset_func: CachedDatasetFunc) -> EvaluationResults:
        """
        Perform a model evaluation by calling repeatedly `.use` on every text of the test dataset and by comparing its resulting category with the expected category.
        """

        evaluated: int = 0

        perfect: int = 0

        squared_error: float = 0.0

        for review in evaluation_dataset_func():
            resulting_category = self.use(review.text)
            log.debug("Evaluation step: %.1d* for %s", resulting_category, review)
            evaluated += 1
            try:
                perfect += 1 if resulting_category == review.rating else 0
                squared_error += (resulting_category - review.rating) ** 2
            except ValueError:
                log.warning("Model execution on %s resulted in a NaN value: %s", review, resulting_category)

        return EvaluationResults(perfect=perfect, evaluated=evaluated, mse=squared_error / evaluated)


@dataclasses.dataclass
class EvaluationResults:
    """
    Container for the results of a dataset evaluation.
    """

    evaluated: int
    """
    The number of reviews that were evaluated.
    """

    perfect: int
    """
    The number of reviews for which the model returned the correct rating.
    """

    mse: float
    """
    Mean squared error
    """

    def __repr__(self):
        return f"<EvaluationResults: {self!s}>"

    def __str__(self):
        return f"Evaluation results:\t{self.evaluated}\tevaluated\t{self.perfect}\tperfect\t{self.perfect / self.evaluated:.2%}\taccuracy\t{self.mse / self.evaluated:.2}\tmean squared error"


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

from __future__ import annotations

import abc
import logging
import collections

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
        er = EvaluationResults()
        for review in evaluation_dataset_func():
            er.add(expected=review.rating, predicted=self.use(review.text))
        return er


class EvaluationResults:
    """
    Container for the results of a dataset evaluation.
    """

    def __init__(self):
        self.confusion_matrix: dict[float, dict[float, int]] = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
        """
        Confusion matrix of the evaluation.

        First key is the expected rating, second key is the output label.
        """

        self.absolute_error_total: float = 0.0
        """
        Sum of the absolute errors committed in the evaluation.
        """

        self.squared_error_total: float = 0.0
        """
        Sum of the squared errors committed in the evaluation.
        """

    def __repr__(self) -> str:
        return f"<EvaluationResults with {self.evaluated_count()} evaluated and {len(self.keys())} categories>"

    def __str__(self) -> str:
        text = [f"Evaluation results: {self.evaluated_count()} evaluated, {self.mean_absolute_error()} mean absolute error, {self.mean_squared_error()} mean squared error, "]
        for key in self.keys():
            text.append(f"{self.recall(key)} recall of {key}, ")
            text.append(f"{self.precision(key)} precision of {key}, ")
        text.append(f"{self.perfect_count()} perfect matches.")
        return "".join(text)

    def __add__(self, other: EvaluationResults) -> EvaluationResults:
        new = self.__class__()
        new.absolute_error_total = self.absolute_error_total + other.absolute_error_total
        new.squared_error_total = self.squared_error_total + other.squared_error_total
        for expected, value in self.confusion_matrix.items():
            for predicted, amount in value.items():
                new.confusion_matrix[expected][predicted] += amount
        for expected, value in other.confusion_matrix.items():
            for predicted, amount in value.items():
                new.confusion_matrix[expected][predicted] += amount
        return new

    def keys(self) -> set[float]:
        """
        Return all processed categories.
        """
        keys: set[float] = set()

        for expected, value in self.confusion_matrix.items():
            keys.add(expected)
            for predicted, _ in value.items():
                keys.add(predicted)

        return keys

    def evaluated_count(self) -> int:
        """
        Return the total number of evaluated reviews.
        """
        total: int = 0
        for row in self.confusion_matrix.values():
            for el in row.values():
                total += el
        return total

    def perfect_count(self) -> int:
        """
        Return the total number of perfect reviews.
        """
        total: int = 0
        for key in self.keys():
            total += self.confusion_matrix[key][key]
        return total

    def recall_count(self, rating: float) -> int:
        """
        Return the number of reviews processed with the given rating.
        """
        total: int = 0
        for el in self.confusion_matrix[rating].values():
            total += el
        return total

    def precision_count(self, rating: float) -> int:
        """
        Return the number of reviews for which the model returned the given rating.
        """
        total: int = 0
        for col in self.confusion_matrix.values():
            total += col[rating]
        return total

    def recall(self, rating: float) -> float:
        """
        Return the recall for a given rating.
        """
        try:
            return self.confusion_matrix[rating][rating] / self.recall_count(rating)
        except KeyError:
            return float("NaN")
        except ZeroDivisionError:
            return float("inf")

    def precision(self, rating: float) -> float:
        """
        Return the precision for a given rating.
        """
        try:
            return self.confusion_matrix[rating][rating] / self.precision_count(rating)
        except KeyError:
            return float("NaN")
        except ZeroDivisionError:
            return float("inf")

    def mean_absolute_error(self) -> float:
        """
        Return the mean absolute error.
        """
        return self.absolute_error_total / self.evaluated_count()

    def mean_squared_error(self) -> float:
        """
        Return the mean squared error.
        """
        return self.squared_error_total / self.evaluated_count()

    def add(self, expected: float, predicted: float) -> None:
        """
        Count a new prediction.
        """
        if expected == predicted:
            log.log(11, "Expected %.1d*, predicted %.1d*", expected, predicted)  # Success
        else:
            log.log(12, "Expected %.1d*, predicted %.1d*", expected, predicted)  # Failure

        self.confusion_matrix[expected][predicted] += 1
        self.absolute_error_total += abs(expected - predicted)
        self.squared_error_total += (expected - predicted) ** 2


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

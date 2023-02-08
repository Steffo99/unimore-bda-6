import abc
import logging
import dataclasses

from ..database import Text, Category, DatasetFunc

log = logging.getLogger(__name__)


@dataclasses.dataclass
class EvaluationResults:
    correct: int
    evaluated: int
    score: float

    def __repr__(self):
        return f"<EvaluationResults: score of {self.score} out of {self.evaluated} evaluated tuples>"

    def __str__(self):
        return f"{self.evaluated} evaluated, {self.correct} correct, {self.correct / self.evaluated * 100:.2} % accuracy, {self.score:.2} score, {self.score / self.evaluated * 100:.2} scoreaccuracy"


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
        score: float = 0.0

        for review in dataset_func():
            resulting_category = self.use(review.text)
            evaluated += 1
            correct += 1 if resulting_category == review.category else 0
            score += 1 - (abs(resulting_category - review.category) / 4)
            if not evaluated % 100:
                temp_results = EvaluationResults(correct=correct, evaluated=evaluated, score=score)
                log.debug(f"{temp_results!s}")

        return EvaluationResults(correct=correct, evaluated=evaluated, score=score)

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

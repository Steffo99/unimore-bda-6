from .base import BaseSentimentAnalyzer
from ..database.cache import CachedDatasetFunc


class ThreeCheat(BaseSentimentAnalyzer):
    """
    A sentiment analyzer that always predicts a 3.0* rating.

    Why? To test the scoring!
    """

    def train(self, training_dataset_func: CachedDatasetFunc, validation_dataset_func: CachedDatasetFunc) -> None:
        pass

    def use(self, text: str) -> float:
        return 3.0


__all__ = (
    "ThreeCheat",
)

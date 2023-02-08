import typing as t
import contextlib
import dataclasses
import logging
import pymongo

from .config import TRAINING_SET_SIZE, VALIDATION_SET_SIZE, EVALUATION_SET_SIZE
from .database import SampleFunc, CachedDatasetFunc, mongo_client_from_config, reviews_collection, store_cache, load_cache, delete_cache

log = logging.getLogger(__name__)


@dataclasses.dataclass
class Caches:
    """
    Container for the three generators that can create datasets.
    """

    training: CachedDatasetFunc
    validation: CachedDatasetFunc
    evaluation: CachedDatasetFunc

    @classmethod
    @contextlib.contextmanager
    def from_database_samples(cls, collection: pymongo.collection.Collection, sample_func: SampleFunc) -> t.ContextManager["Caches"]:
        """
        Create a new caches object from a database collection and a sampling function.
        """

        log.debug("Gathering datasets...")
        reviews_training = sample_func(collection, TRAINING_SET_SIZE.__wrapped__)
        reviews_validation = sample_func(collection, VALIDATION_SET_SIZE.__wrapped__)
        reviews_evaluation = sample_func(collection, EVALUATION_SET_SIZE.__wrapped__)

        log.debug("Caching datasets...")
        store_cache(reviews_training, "./data/training")
        store_cache(reviews_validation, "./data/validation")
        store_cache(reviews_evaluation, "./data/evaluation")

        log.debug("Loading dataset caches...")
        training_cache = load_cache("./data/training")
        validation_cache = load_cache("./data/validation")
        evaluation_cache = load_cache("./data/evaluation")

        yield Caches(training=training_cache, validation=validation_cache, evaluation=evaluation_cache)

        log.debug("Cleaning up caches...")
        delete_cache("./data/training")
        delete_cache("./data/validation")
        delete_cache("./data/evaluation")

    @staticmethod
    def ensure_clean():
        log.debug("Ensuring there are no leftover caches...")

        try:
            delete_cache("./data/training")
        except FileNotFoundError:
            pass

        try:
            delete_cache("./data/validation")
        except FileNotFoundError:
            pass

        try:
            delete_cache("./data/evaluation")
        except FileNotFoundError:
            pass


__all__ = (
    "Caches",
)

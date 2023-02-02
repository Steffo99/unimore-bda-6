import typing as t
import pymongo
import pymongo.collection
import contextlib
import bson
import logging

from .config import MONGO_HOST, MONGO_PORT, WORKING_SET_SIZE, TRAINING_SET_SIZE, TEST_SET_SIZE

log = logging.getLogger(__name__)


class Review(t.TypedDict):
    _id: bson.ObjectId
    reviewerID: str
    asin: str
    reviewerName: str
    helpful: tuple[int, int]
    reviewText: str
    overall: float
    summary: str
    unixReviewTime: int
    reviewTime: str


@contextlib.contextmanager
def mongo_client_from_config() -> t.ContextManager[pymongo.MongoClient]:
    """
    Create a new MongoDB client and yield it.
    """
    log.debug("Opening connection to MongoDB...")
    client = pymongo.MongoClient(
        host=MONGO_HOST.__wrapped__,
        port=MONGO_PORT.__wrapped__,
    )
    log.info("Opened connection to MongoDB: %s", client)

    yield client

    log.info("Closing connection to MongoDB: %s", client)
    client.close()
    log.debug("Closed connection to MongoDB!")


@contextlib.contextmanager
def mongo_reviews_collection_from_config() -> pymongo.collection.Collection[Review]:
    """
    Create a new MongoDB client, access the ``reviews`` collection in the ``reviews`` database, and yield it.
    """
    with mongo_client_from_config() as db:
        log.debug("Accessing the reviews collection...")
        collection = db.reviews.reviews
        log.debug("Collection accessed successfully: %s", collection)
        yield collection


def sample_reviews(reviews: pymongo.collection.Collection, amount: int) -> t.Iterable[Review]:
    """
    Get ``amount`` random reviews from the ``reviews`` collection.
    """
    log.debug("Getting a sample of %d reviews...", amount)

    return reviews.aggregate([
        {"$limit": WORKING_SET_SIZE.__wrapped__},
        {"$sample": {"size": amount}},
    ])


def sample_reviews_by_rating(reviews: pymongo.collection.Collection, rating: float, amount: int) -> t.Iterable[Review]:
    """
    Get ``amount`` random reviews with ``rating`` stars from the ``reviews`` collection.
    """
    log.debug("Getting a sample of %d reviews with %d stars...", amount, rating)

    return reviews.aggregate([
        {"$limit": WORKING_SET_SIZE.__wrapped__},
        {"$match": {"overall": rating}},
        {"$sample": {"size": amount}},
    ])


def sample_reviews_by_rating_polar(reviews: pymongo.collection.Collection, amount: int) -> t.Iterable[Review]:
    """
    Get ``amount`` random reviews with either a 5-star or 1-star rating from the ``reviews`` collection.
    """
    log.debug("Getting a sample of %d reviews with either 5 or 1 stars...", amount)

    return reviews.aggregate([
        {"$limit": WORKING_SET_SIZE.__wrapped__},
        {"$match":
            {"$or":
                [
                    {"overall": 1.0},
                    {"overall": 5.0},
                ]
            },
        },
        {"$sample": {"size": amount}},
    ])


def get_training_reviews(collection: pymongo.collection.Collection) -> list[Review]:
    """
    Get the subset of reviews that should act as training set.
    """
    log.info("Building training set...")

    # Get the amount from the config
    amount: int = TRAINING_SET_SIZE.__wrapped__

    # Handle odd numbers
    positive_amount: int = amount // 2
    negative_amount: int = amount - positive_amount

    # Sample the required reviews
    positive = sample_reviews_by_rating(collection, 5.0, positive_amount)
    negative = sample_reviews_by_rating(collection, 1.0, negative_amount)

    # Randomness here does not matter, so just merge the lists
    both = [*positive, *negative]

    return both


def get_test_reviews(collection: pymongo.collection.Collection) -> list[Review]:
    """
    Get the subset of reviews that should act as test set.
    """

    log.info("Building test set...")

    amount: int = TEST_SET_SIZE.__wrapped__

    return list(sample_reviews_by_rating_polar(collection, amount))


__all__ = (
    "Review",
    "mongo_client_from_config",
    "mongo_reviews_collection_from_config",
    "sample_reviews",
    "sample_reviews_by_rating",
    "sample_reviews_by_rating_polar",
    "get_training_reviews",
    "get_test_reviews",
)

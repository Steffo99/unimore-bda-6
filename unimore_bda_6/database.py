import typing as t
import pymongo
import pymongo.collection
import contextlib
import bson
import logging
import itertools

from .config import MONGO_HOST, MONGO_PORT, WORKING_SET_SIZE, DATA_SET_SIZE

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
    client: pymongo.MongoClient = pymongo.MongoClient(
        host=MONGO_HOST.__wrapped__,
        port=MONGO_PORT.__wrapped__,
    )
    log.info("Opened connection to MongoDB at %s!", client.address)

    yield client

    log.info("Closing connection to MongoDB...")
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


def sample_reviews(reviews: pymongo.collection.Collection, amount: int) -> t.Iterator[Review]:
    """
    Get ``amount`` random reviews from the ``reviews`` collection.
    """
    log.debug("Getting a sample of %d reviews...", amount)

    return reviews.aggregate([
        {"$limit": WORKING_SET_SIZE.__wrapped__},
        {"$sample": {"size": amount}},
    ])


def sample_reviews_by_rating(reviews: pymongo.collection.Collection, rating: float, amount: int) -> t.Iterator[Review]:
    """
    Get ``amount`` random reviews with ``rating`` stars from the ``reviews`` collection.
    """
    log.debug("Getting a sample of %d reviews with %d stars...", amount, rating)

    return reviews.aggregate([
        {"$limit": WORKING_SET_SIZE.__wrapped__},
        {"$match": {"overall": rating}},
        {"$sample": {"size": amount}},
    ])


def dataset_polar(collection: pymongo.collection.Collection, amount: int) -> t.Iterator[Review]:
    """
    Get a list of the same amount of 1-star and 5-star reviews.
    """
    log.info("Building polar dataset with %d reviews...", amount * 2)

    # Sample the required reviews
    positive = sample_reviews_by_rating(collection, rating=5.0, amount=amount)
    negative = sample_reviews_by_rating(collection, rating=1.0, amount=amount)

    # Randomness here does not matter, so just merge the lists
    both = itertools.chain(positive, negative)

    return both


def dataset_varied(collection: pymongo.collection.Collection, amount: int) -> t.Iterator[Review]:
    """
    Get a list of the same amount of reviews for each rating.
    """
    log.info("Building varied dataset with %d reviews...", amount * 5)

    # Sample the required reviews
    terrible = sample_reviews_by_rating(collection, rating=1.0, amount=amount)
    negative = sample_reviews_by_rating(collection, rating=2.0, amount=amount)
    mixed    = sample_reviews_by_rating(collection, rating=3.0, amount=amount)
    positive = sample_reviews_by_rating(collection, rating=4.0, amount=amount)
    great    = sample_reviews_by_rating(collection, rating=5.0, amount=amount)

    full = itertools.chain(terrible, negative, mixed, positive, great)

    return full


__all__ = (
    "Review",
    "mongo_client_from_config",
    "mongo_reviews_collection_from_config",
    "sample_reviews",
    "sample_reviews_by_rating",
    "dataset_polar",
    "dataset_varied",
)

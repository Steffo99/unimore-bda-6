import typing as t
import pymongo
import pymongo.collection
import contextlib
import bson
import logging
import itertools
import collections

from .config import MONGO_HOST, MONGO_PORT, WORKING_SET_SIZE

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


Text = str
Category = float
DataTuple = collections.namedtuple("DataTuple", ["text", "category"], verbose=True)
DataSet = t.Iterable[DataTuple]


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


def review_to_datatuple(review: Review) -> DataTuple:
    """
    Return the label corresponding to the given review.

    Possible categories are:

    * terrible (1.0)
    * negative (2.0)
    * mixed (3.0)
    * positive (4.0)
    * great (5.0)
    * unknown (everything else)
    """
    text = review["reviewText"]
    category = review["overall"]

    return DataTuple(text=text, category=category)


def polar_dataset(collection: pymongo.collection.Collection, amount: int) -> t.Iterator[DataTuple]:
    """
    Get a list of the same amount of 1-star and 5-star reviews.
    """
    log.info("Building polar dataset with %d reviews...", amount * 2)

    # Sample the required reviews
    positive = sample_reviews_by_rating(collection, rating=5.0, amount=amount)
    negative = sample_reviews_by_rating(collection, rating=1.0, amount=amount)

    # Chain the iterators
    full = itertools.chain(positive, negative)

    # Convert reviews to datatuples
    full = map(review_to_datatuple, full)

    return full


def varied_dataset(collection: pymongo.collection.Collection, amount: int) -> t.Iterator[DataTuple]:
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

    # Chain the iterators
    full = itertools.chain(terrible, negative, mixed, positive, great)

    # Convert reviews to datatuples
    full = map(review_to_datatuple, full)

    return full


__all__ = (
    "Review",
    "Text",
    "Category",
    "DataTuple",
    "DataSet",
    "mongo_client_from_config",
    "mongo_reviews_collection_from_config",
    "sample_reviews",
    "sample_reviews_by_rating",
    "polar_dataset",
    "varied_dataset",
)

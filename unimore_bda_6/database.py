import typing as t
import pymongo
import pymongo.collection
import contextlib
import bson
import logging
import tensorflow

from .config import MONGO_HOST, MONGO_PORT, WORKING_SET_SIZE

log = logging.getLogger(__name__)


class MongoReview(t.TypedDict):
    """
    A review as it is stored on MongoDB.

    .. warning:: Do not instantiate: this is only for type hints!
    """
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


class Review:
    def __init__(self, text: Text, category: Category):
        self.text: Text = text
        self.category: Category = category

    @classmethod
    def from_mongoreview(cls, review: MongoReview):
        return cls(
            text=review["reviewText"],
            category=review["overall"],
        )

    def __repr__(self):
        return f"<{self.__class__.__qualname__}: [{self.category}] {self.text}>"

    def __getitem__(self, item):
        if item == 0 or item == "text":
            return self.text
        elif item == 1 or item == "category":
            return self.category
        else:
            raise KeyError(item)

    def to_tensor_tuple(self) -> tuple[tensorflow.Tensor, tensorflow.Tensor]:
        return tensorflow.convert_to_tensor(self.text, dtype=tensorflow.string), tensorflow.convert_to_tensor(self.category, dtype=tensorflow.string)


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
    log.info("Opened connection to MongoDB!")

    yield client

    log.info("Closing connection to MongoDB...")
    client.close()
    log.debug("Closed connection to MongoDB!")


@contextlib.contextmanager
def mongo_reviews_collection_from_config() -> pymongo.collection.Collection[MongoReview]:
    """
    Create a new MongoDB client, access the ``reviews`` collection in the ``reviews`` database, and yield it.
    """
    with mongo_client_from_config() as db:
        log.debug("Accessing the reviews collection...")
        collection = db.reviews.reviews
        log.debug("Collection accessed successfully: %s", collection)
        yield collection


class DatasetFunc(t.Protocol):
    def __call__(self) -> t.Iterator[Review]:
        pass


def sample_reviews(collection: pymongo.collection.Collection, amount: int) -> t.Iterator[Review]:
    """
    Get ``amount`` random reviews from the ``reviews`` collection.
    """
    log.debug("Getting a sample of %d reviews...", amount)

    cursor = collection.aggregate([
        {"$limit": WORKING_SET_SIZE.__wrapped__},
        {"$sample": {"size": amount}},
    ])

    cursor = map(Review.from_mongoreview, cursor)
    return cursor


def sample_reviews_by_rating(collection: pymongo.collection.Collection, rating: float, amount: int) -> t.Iterator[Review]:
    """
    Get ``amount`` random reviews with ``rating`` stars from the ``reviews`` collection.
    """
    log.debug("Getting a sample of %d reviews with %d stars...", amount, rating)

    cursor = collection.aggregate([
        {"$limit": WORKING_SET_SIZE.__wrapped__},
        {"$match": {"overall": rating}},
        {"$sample": {"size": amount}},
    ])

    cursor = map(Review.from_mongoreview, cursor)
    return cursor


def sample_reviews_polar(collection: pymongo.collection.Collection, amount: int) -> t.Iterator[Review]:
    log.debug("Getting a sample of %d polar reviews...", amount * 2)

    cursor = collection.aggregate([
        {"$limit": WORKING_SET_SIZE.__wrapped__},
        {"$match": {"overall": 1.0}},
        {"$sample": {"size": amount}},
        {"$unionWith": {
            "coll": collection.name,
            "pipeline": [
                {"$limit": WORKING_SET_SIZE.__wrapped__},
                {"$match": {"overall": 5.0}},
                {"$sample": {"size": amount}},
            ],
        }}
    ])

    cursor = map(Review.from_mongoreview, cursor)
    return cursor


def sample_reviews_varied(collection: pymongo.collection.Collection, amount: int) -> t.Iterator[Review]:
    log.debug("Getting a sample of %d varied reviews...", amount * 5)

    # Wow, this is ugly.
    cursor = collection.aggregate([
        {"$limit": WORKING_SET_SIZE.__wrapped__},
        {"$match": {"overall": 1.0}},
        {"$sample": {"size": amount}},
        {"$unionWith": {
            "coll": collection.name,
            "pipeline": [
                {"$limit": WORKING_SET_SIZE.__wrapped__},
                {"$match": {"overall": 2.0}},
                {"$sample": {"size": amount}},
                {"$unionWith": {
                    "coll": collection.name,
                    "pipeline": [
                        {"$limit": WORKING_SET_SIZE.__wrapped__},
                        {"$match": {"overall": 3.0}},
                        {"$sample": {"size": amount}},
                        {"$unionWith": {
                            "coll": collection.name,
                            "pipeline": [
                                {"$limit": WORKING_SET_SIZE.__wrapped__},
                                {"$match": {"overall": 4.0}},
                                {"$sample": {"size": amount}},
                                {"$unionWith": {
                                    "coll": collection.name,
                                    "pipeline": [
                                        {"$limit": WORKING_SET_SIZE.__wrapped__},
                                        {"$match": {"overall": 5.0}},
                                        {"$sample": {"size": amount}},
                                    ],
                                }}
                            ],
                        }}
                    ],
                }}
            ],
        }}
    ])

    cursor = map(Review.from_mongoreview, cursor)
    return cursor


__all__ = (
    "Text",
    "Category",
    "Review",
    "DatasetFunc",
    "mongo_client_from_config",
    "mongo_reviews_collection_from_config",
    "sample_reviews",
    "sample_reviews_by_rating",
    "sample_reviews_polar",
    "sample_reviews_varied",
)

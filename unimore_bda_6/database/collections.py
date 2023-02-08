import pymongo.collection
import typing as t
import bson
import logging

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


def reviews_collection(db: pymongo.MongoClient) -> pymongo.collection.Collection[MongoReview]:
    """
    Create a new MongoDB client, access the ``reviews`` collection in the ``reviews`` database, and yield it.
    """
    log.debug("Accessing the reviews collection...")
    collection: pymongo.collection.Collection[MongoReview] = db.reviews.reviews
    log.debug("Collection accessed successfully: %s", collection.name)
    return collection


__all__ = (
    "MongoReview",
    "reviews_collection",
)

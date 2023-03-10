import logging
import pymongo
import typing as t

from ..config import WORKING_SET_SIZE
from .datatypes import TextReview

log = logging.getLogger(__name__)


SampleFunc = t.Callable[[pymongo.collection.Collection, int], t.Iterator[TextReview]]


def sample_reviews(collection: pymongo.collection.Collection, amount: int) -> t.Iterator[TextReview]:
    """
    Get ``amount`` random reviews from the ``reviews`` collection.
    """
    log.debug("Getting a sample of %d reviews...", amount)

    cursor = collection.aggregate([
        {"$limit": WORKING_SET_SIZE.__wrapped__},
        {"$sample": {"size": amount}},
    ])

    cursor = map(TextReview.from_mongoreview, cursor)

    return cursor


def sample_reviews_by_rating(collection: pymongo.collection.Collection, rating: float, amount: int) -> t.Iterator[TextReview]:
    """
    Get ``amount`` random reviews with ``rating`` stars from the ``reviews`` collection.
    """
    log.debug("Getting a sample of %d reviews with %d stars...", amount, rating)

    cursor = collection.aggregate([
        {"$limit": WORKING_SET_SIZE.__wrapped__},
        {"$match": {"overall": rating}},
        {"$sample": {"size": amount}},
    ])

    return cursor


def sample_reviews_polar(collection: pymongo.collection.Collection, amount: int) -> t.Iterator[TextReview]:
    category_amount = amount // 2

    log.debug("Getting a sample of %d polar reviews...", category_amount * 2)

    cursor = collection.aggregate([
        {"$limit": WORKING_SET_SIZE.__wrapped__},
        {"$match": {"overall": 1.0}},
        {"$sample": {"size": category_amount}},
        {"$unionWith": {
            "coll": collection.name,
            "pipeline": [
                {"$limit": WORKING_SET_SIZE.__wrapped__},
                {"$match": {"overall": 5.0}},
                {"$sample": {"size": category_amount}},
            ],
        }},
        {"$addFields": {
            "sortKey": {"$rand": {}},
        }},
        {"$sort": {
            "sortKey": 1,
        }}
    ])

    cursor = map(TextReview.from_mongoreview, cursor)

    return cursor


def sample_reviews_varied(collection: pymongo.collection.Collection, amount: int) -> t.Iterator[TextReview]:
    category_amount = amount // 5

    log.debug("Getting a sample of %d varied reviews...", category_amount * 5)

    cursor = collection.aggregate([
        {"$limit": WORKING_SET_SIZE.__wrapped__},
        {"$match": {"overall": 1.0}},
        {"$sample": {"size": category_amount}},
        {"$unionWith": {
            "coll": collection.name,
            "pipeline": [
                {"$limit": WORKING_SET_SIZE.__wrapped__},
                {"$match": {"overall": 2.0}},
                {"$sample": {"size": category_amount}},
                {"$unionWith": {
                    "coll": collection.name,
                    "pipeline": [
                        {"$limit": WORKING_SET_SIZE.__wrapped__},
                        {"$match": {"overall": 3.0}},
                        {"$sample": {"size": category_amount}},
                        {"$unionWith": {
                            "coll": collection.name,
                            "pipeline": [
                                {"$limit": WORKING_SET_SIZE.__wrapped__},
                                {"$match": {"overall": 4.0}},
                                {"$sample": {"size": category_amount}},
                                {"$unionWith": {
                                    "coll": collection.name,
                                    "pipeline": [
                                        {"$limit": WORKING_SET_SIZE.__wrapped__},
                                        {"$match": {"overall": 5.0}},
                                        {"$sample": {"size": category_amount}},
                                    ],
                                }}
                            ],
                        }}
                    ],
                }}
            ],
        }},
        {"$addFields": {
            "sortKey": {"$rand": {}},
        }},
        {"$sort": {
            "sortKey": 1,
        }}
    ])

    cursor = map(TextReview.from_mongoreview, cursor)

    return cursor


__all__ = (
    "SampleFunc",
    "sample_reviews",
    "sample_reviews_by_rating",
    "sample_reviews_polar",
    "sample_reviews_varied",
)

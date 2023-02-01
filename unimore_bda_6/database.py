import typing as t
import pymongo
import pymongo.collection
import contextlib
import bson

from .config import MONGO_HOST, MONGO_PORT


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
    client = pymongo.MongoClient(
        host=MONGO_HOST.__resolved__,
        port=MONGO_PORT.__resolved__,
    )
    yield client
    client.close()


@contextlib.contextmanager
def mongo_reviews_collection_from_config() -> pymongo.collection.Collection[Review]:
    """
    Create a new MongoDB client, access the ``reviews`` collection in the ``reviews`` database, and yield it.
    """
    with mongo_client_from_config() as db:
        yield db.reviews.reviews


def sample_reviews(reviews: pymongo.collection.Collection, amount: int) -> t.Iterable[Review]:
    """
    Get ``amount`` random reviews from the ``reviews`` collection.
    """

    return reviews.aggregate([
        {"$sample": {"size": amount}}
    ])


def sample_reviews_by_rating(reviews: pymongo.collection.Collection, rating: float, amount: int) -> t.Iterable[Review]:
    """
    Get ``amount`` random reviews with ``rating`` stars from the ``reviews`` collection.
    """

    return reviews.aggregate([
        {"$match": {"overall": rating}},
        {"$sample": {"size": amount}},
    ])


def sample_reviews_by_rating_polar(reviews: pymongo.collection.Collection, amount: int) -> t.Iterable[Review]:
    """
    Get ``amount`` random reviews with either a 5-star or 1-star rating from the ``reviews`` collection.
    """

    return reviews.aggregate([
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


def get_reviews_training_set(reviews: pymongo.collection.Collection, amount: int) -> t.Iterable[Review]:
    """
    Get the subset of reviews that should act as training set.
    """

    # Handle odd numbers
    positive_amount: int = amount // 2
    negative_amount: int = amount - positive_amount

    # Sample the required reviews
    positive = sample_reviews_by_rating(reviews, 5.0, positive_amount)
    negative = sample_reviews_by_rating(reviews, 1.0, negative_amount)

    # Randomness here does not matter, so just merge the lists
    both = [*positive, *negative]

    return both


def get_reviews_test_set(reviews: pymongo.collection.Collection, amount: int) -> t.Iterable[Review]:
    """
    Get the subset of reviews that should act as test set.
    """

    return sample_reviews_by_rating_polar(reviews, amount)

import abc
import typing as t

from .collections import MongoReview


class Review(metaclass=abc.ABCMeta):
    """
    Base class for method common to both review types.
    """

    def __init__(self, *, rating: float):
        self.rating: float = rating
        """
        The star rating of the review, from ``1.0`` to ``5.0``.
        """


class TextReview(Review):
    """
    Optimized container for a review with the text still intact.

    Uses `__slots__` for better performance.
    """

    __slots__ = (
        "text",
        "rating",
    )

    def __init__(self, *, rating: float, text: str):
        super().__init__(rating=rating)

        self.text: str = text
        """
        The contents of the review.
        """

    @classmethod
    def from_mongoreview(cls, review: MongoReview) -> "TextReview":
        """
        Create a new `.Review` object from a `MongoReview` `dict`.
        """
        return cls(
            text=review["reviewText"],
            rating=review["overall"],
        )

    def __repr__(self):
        return f"<{self.__class__.__qualname__}: ({self.rating}*) {self.text[:80]}>"


class TokenizedReview(Review):
    """
    Optimized container for a review with a tokenized text.

    Uses `__slots__` for better performance.
    """

    __slots__ = (
        "tokens",
        "rating",
    )

    def __init__(self, *, rating: float, tokens: t.Iterator[str]):
        super().__init__(rating=rating)

        self.tokens: list[str] = list(tokens)
        """
        List of all tokens in the review text.
        """

    def __repr__(self):
        return f"<{self.__class__.__qualname__}: ({self.rating}*) [{len(self.tokens)} tokens]>"


__all__ = (
    "TextReview",
    "TokenizedReview",
)

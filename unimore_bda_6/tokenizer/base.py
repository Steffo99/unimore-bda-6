import typing as t
import abc
from ..database.datatypes import TextReview, TokenizedReview


class BaseTokenizer(metaclass=abc.ABCMeta):
    """
    The base for all tokenizers in this project.
    """

    def __repr__(self):
        return f"<{self.__class__.__qualname__}>"

    @abc.abstractmethod
    def tokenize(self, text: str) -> t.Iterator[str]:
        """
        Convert a text `str` into another `str` containing a series of whitespace-separated tokens.
        """
        raise NotImplementedError()

    def tokenize_review(self, review: TextReview) -> TokenizedReview:
        """
        Apply `.tokenize` to the text of a `TextReview`, converting it in a `TokenizedReview`.
        """
        tokens = self.tokenize(review.text)
        return TokenizedReview(rating=review.rating, tokens=tokens)

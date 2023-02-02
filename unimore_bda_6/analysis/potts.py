from ..vendor.potts import Tokenizer
from .vanilla import VanillaSA, VanillaReviewSA, VanillaUniformReviewSA


class PottsSA(VanillaSA):
    """
    A sentiment analyzer using Potts' tokenizer.
    """

    def __init__(self) -> None:
        super().__init__()

    def _tokenize_text(self, text: str) -> list[str]:
        """
        Convert a text string into a list of tokens, using the language of the model.
        """
        tokenizer: Tokenizer = Tokenizer(preserve_case=False)
        return list(tokenizer.tokenize(text))


class PottsReviewSA(VanillaReviewSA, PottsSA):
    """
    A `PottsSA` to be used with `Review`s.
    """


class PottsUniformReviewSA(VanillaUniformReviewSA, PottsSA):
    """
    A `PottsSA` with 5 buckets instead of 2.
    """


__all__ = (
    "PottsSA",
    "PottsReviewSA",
)
from . import nltk_based
from . import potts_based


all_tokenizers = [
    nltk_based.tokenizer,
    potts_based.tokenizer,
]


__all__ = (
    "nltk_based",
    "potts_based",
    "all_tokenizers",
)

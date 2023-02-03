from . import nltk_based
from . import potts_based


all_tokenizers = [
    nltk_based.nltk_tokenizer,
    potts_based.potts_tokenizer,
]


__all__ = (
    "nltk_based",
    "potts_based",
    "all_tokenizers",
)

import nltk
import nltk.sentiment.util


def tokenizer(text: str) -> list[str]:
    """
    Convert a text string into a list of tokens.
    """
    tokens = nltk.word_tokenize(text)
    nltk.sentiment.util.mark_negation(tokens, shallow=True)
    return tokens


__all__ = (
    "tokenizer",
)

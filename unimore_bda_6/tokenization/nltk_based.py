import nltk
import nltk.sentiment.util


def nltk_tokenizer(text: str) -> list[str]:
    """
    Convert a text string into a list of tokens.
    """
    tokens = nltk.word_tokenize(text)
    nltk.sentiment.util.mark_negation(tokens, shallow=True)
    return tokens


__all__ = (
    "nltk_tokenizer",
)

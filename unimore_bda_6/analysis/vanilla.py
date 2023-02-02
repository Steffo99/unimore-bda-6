import abc
import nltk
import nltk.classify
import nltk.sentiment
import nltk.sentiment.util
import logging
import typing as t

from ..database import Review
from .base import BaseSA, AlreadyTrainedError, NotTrainedError


log = logging.getLogger(__name__)


class VanillaSA(BaseSA, metaclass=abc.ABCMeta):
    """
    A sentiment analyzer resembling the one implemented in structure the one implemented in the classroom, using the basic sentiment analyzer of NLTK.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model: nltk.sentiment.SentimentAnalyzer = nltk.sentiment.SentimentAnalyzer()

    def _tokenize_text(self, text: str) -> list[str]:
        """
        Convert a text string into a list of tokens.
        """
        tokens = nltk.word_tokenize(text)
        nltk.sentiment.util.mark_negation(tokens, shallow=True)
        return tokens

    def __add_feature_unigrams(self, training_set: list[tuple[list[str], str]]) -> None:
        """
        Add the `nltk.sentiment.util.extract_unigram_feats` feature to the model.
        """
        all_words = self.model.all_words(training_set, labeled=True)
        unigrams = self.model.unigram_word_feats(words=all_words, min_freq=4)
        self.model.add_feat_extractor(nltk.sentiment.util.extract_unigram_feats, unigrams=unigrams)

    def _featurize_documents(self, documents: list[tuple[list[str], str]]):
        """
        Apply features to a document.
        """
        return self.model.apply_features(documents, labeled=True)

    def _train_with_set(self, training_set: list[tuple[list[str], str]]) -> None:
        """
        Train the model with the given **pre-classified but not pre-tokenized** training set.
        """
        if self.trained:
            raise AlreadyTrainedError()

        self.__add_feature_unigrams(training_set)
        training_set_with_features = self._featurize_documents(training_set)

        self.model.train(trainer=nltk.classify.NaiveBayesClassifier.train, training_set=training_set_with_features)
        self.trained = True

    def _evaluate_with_set(self, test_set: list[tuple[list[str], str]]) -> dict:
        if not self.trained:
            raise NotTrainedError()
        
        test_set_with_features = self._featurize_documents(test_set)
        return self.model.evaluate(test_set_with_features)

    def _use_with_tokens(self, tokens: list[str]) -> str:
        if not self.trained:
            raise NotTrainedError()
        
        return self.model.classify(instance=tokens)


class VanillaReviewSA(VanillaSA):
    """
    A `VanillaSA` to be used with `Review`s.
    """

    def __init__(self, categorizer: t.Callable[[Review], str]) -> None:
        super().__init__()
        self.categorizer: t.Callable[[Review], str] = categorizer

    def _review_to_data_set(self, review: Review) -> tuple[list[str], str]:
        """
        Convert a review to a NLTK-compatible dataset.
        """
        return self._tokenize_text(text=review["reviewText"]), self.categorizer(rating=review["overall"])
        
    def train(self, reviews: t.Iterable[Review]) -> None:
        data_set = list(map(self._review_to_data_set, reviews))
        self._train_with_set(data_set)

    def evaluate(self, reviews: t.Iterable[Review]):
        data_set = list(map(self._review_to_data_set, reviews))
        return self._evaluate_with_set(data_set)

    def use(self, text: str) -> str:
        return self._use_with_tokens(self._tokenize_text(text))


def polar_categorizer(rating: float) -> str:
    """
    Return the polar label corresponding to the given rating.

    Possible categories are:
    
    * negative (1.0, 2.0)
    * positive (3.0, 4.0, 5.0)
    * unknown (everything else)
    """
    match rating:
        case 1.0 | 2.0:
            return "negative"
        case 3.0 | 4.0 | 5.0:
            return "positive"
        case _:
            return "unknown"


def stars_categorizer(rating: float) -> str:
    """
    Return the "stars" label corresponding to the given rating.

    Possible categories are:
    
    * terrible (1.0)
    * negative (2.0)
    * mixed (3.0)
    * positive (4.0)
    * great (5.0)
    * unknown (everything else)
    """
    match rating:
        case 1.0:
            return "terrible"
        case 2.0:
            return "negative"
        case 3.0:
            return "mixed"
        case 4.0:
            return "positive"
        case 5.0:
            return "great"
        case _:
            return "unknown"


__all__ = (
    "VanillaSA",
    "VanillaReviewSA",
    "polar_categorizer",
    "stars_categorizer",
)

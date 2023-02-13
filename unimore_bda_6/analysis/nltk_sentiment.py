import nltk
import nltk.classify
import nltk.sentiment
import nltk.sentiment.util
import logging
import typing as t
import itertools

from ..database import TextReview, CachedDatasetFunc, TokenizedReview
from .base import BaseSentimentAnalyzer, AlreadyTrainedError, NotTrainedError
from ..log import count_passage
from ..tokenizer import BaseTokenizer

log = logging.getLogger(__name__)

TokenBag = list[str]
Features = dict[str, int]


class NLTKSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    A sentiment analyzer resembling the one implemented in structure the one implemented in the classroom, using the basic sentiment analyzer of NLTK.
    """

    def __init__(self, *, tokenizer: BaseTokenizer) -> None:
        super().__init__(tokenizer=tokenizer)

        self.model: nltk.sentiment.SentimentAnalyzer = nltk.sentiment.SentimentAnalyzer()
        self.trained: bool = False

    def _add_feature_unigrams(self, dataset: t.Iterator[TokenizedReview]) -> None:
        """
        Register the `nltk.sentiment.util.extract_unigram_feats` feature extrator on the model.
        """
        # Ignore the category and only access the tokens
        tokenbags = map(lambda r: r.tokens, dataset)
        # Get all words in the documents
        all_words = self.model.all_words(tokenbags, labeled=False)
        # Create unigram `contains(*)` features from the previously gathered words
        unigrams = self.model.unigram_word_feats(words=all_words, min_freq=4)
        # Add the feature extractor to the model
        self.model.add_feat_extractor(nltk.sentiment.util.extract_unigram_feats, unigrams=unigrams)

    def _add_feature_extractors(self, dataset: t.Iterator[TextReview]):
        """
        Register new feature extractors on the `.model`.
        """
        # Tokenize the reviews
        dataset: t.Iterator[TokenizedReview] = map(self.tokenizer.tokenize_review, dataset)
        # Add the unigrams feature
        self._add_feature_unigrams(dataset)

    def __extract_features(self, review: TextReview) -> tuple[Features, float]:
        """
        Convert a (TokenBag, Category) tuple to a (Features, Category) tuple.

        Does not use `SentimentAnalyzer.apply_features` due to unexpected behaviour when using iterators.
        """
        review: TokenizedReview = self.tokenizer.tokenize_review(review)
        return self.model.extract_features(review.tokens), review.rating

    def train(self, training_dataset_func: CachedDatasetFunc, validation_dataset_func: CachedDatasetFunc) -> None:
        # Forbid retraining the model
        if self.trained:
            raise AlreadyTrainedError()

        # Add the feature extractors to the model
        self._add_feature_extractors(training_dataset_func())

        # Extract features from the dataset
        featureset: t.Iterator[tuple[Features, float]] = map(self.__extract_features, training_dataset_func())

        # Train the classifier with the extracted features and category
        self.model.classifier = nltk.classify.NaiveBayesClassifier.train(featureset)

        # Toggle the trained flag
        self.trained = True

    def use(self, text: str) -> float:
        # Require the model to be trained
        if not self.trained:
            raise NotTrainedError()

        # Tokenize the input
        tokens = self.tokenizer.tokenize(text)

        # Run the classification method
        return self.model.classify(instance=tokens)


__all__ = (
    "NLTKSentimentAnalyzer",
)

import nltk
import nltk.classify
import nltk.sentiment
import nltk.sentiment.util
import logging
import typing as t
import itertools

from ..database import Text, Category, DataTuple, DataSet
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

    def __tokenize_datatuple(self, datatuple: DataTuple) -> tuple[TokenBag, Category]:
        """
        Convert the `Text` of a `DataTuple` to a `TokenBag`.
        """
        count_passage(log, "tokenize_datatuple", 100)
        return self.tokenizer.tokenize_builtins(datatuple[0]), datatuple[1]

    def _add_feature_unigrams(self, dataset: t.Iterator[tuple[TokenBag, Category]]) -> None:
        """
        Register the `nltk.sentiment.util.extract_unigram_feats` feature extrator on the model.
        """
        # Ignore the category and only access the tokens
        tokenbags = map(lambda d: d[0], dataset)
        # Get all words in the documents
        all_words = self.model.all_words(tokenbags, labeled=False)
        # Create unigram `contains(*)` features from the previously gathered words
        unigrams = self.model.unigram_word_feats(words=all_words, min_freq=4)
        # Add the feature extractor to the model
        self.model.add_feat_extractor(nltk.sentiment.util.extract_unigram_feats, unigrams=unigrams)

    def _add_feature_extractors(self, dataset: t.Iterator[tuple[TokenBag, Category]]):
        """
        Register new feature extractors on the `.model`.
        """
        # Add the unigrams feature
        self._add_feature_unigrams(dataset)

    def __extract_features(self, data: tuple[TokenBag, Category]) -> tuple[Features, Category]:
        """
        Convert a (TokenBag, Category) tuple to a (Features, Category) tuple.

        Does not use `SentimentAnalyzer.apply_features` due to unexpected behaviour when using iterators.
        """
        count_passage(log, "extract_features", 100)
        return self.model.extract_features(data[0]), data[1]

    def train(self, dataset: DataSet) -> None:
        # Forbid retraining the model
        if self.trained:
            raise AlreadyTrainedError()

        # Tokenize the dataset
        dataset: t.Iterator[tuple[TokenBag, Category]] = map(self.__tokenize_datatuple, dataset)

        # Cleanly duplicate the dataset iterator
        # Reduce average memory footprint, but not maximum
        dataset_1, dataset_2 = itertools.tee(dataset, 2)
        dataset_1: t.Iterator[tuple[TokenBag, Category]]
        dataset_2: t.Iterator[tuple[TokenBag, Category]]

        # Add the feature extractors to the model
        self._add_feature_extractors(dataset_1)
        del dataset_1  # Delete exausted iterator

        # Extract features from the dataset
        dataset_2: t.Iterator[tuple[Features, Category]] = map(self.__extract_features, dataset_2)

        # Train the classifier with the extracted features and category
        self.model.classifier = nltk.classify.NaiveBayesClassifier.train(dataset_2)

        # Toggle the trained flag
        self.trained = True

    def use(self, text: Text) -> Category:
        # Require the model to be trained
        if not self.trained:
            raise NotTrainedError()

        # Tokenize the input
        tokens = self.tokenizer.tokenize_builtins(text)

        # Run the classification method
        return self.model.classify(instance=tokens)


__all__ = (
    "NLTKSentimentAnalyzer",
)

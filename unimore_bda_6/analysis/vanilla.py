import nltk
import nltk.classify
import nltk.sentiment
import nltk.sentiment.util
import logging
import typing as t
import itertools

from .base import Input, Category, BaseSA, AlreadyTrainedError, NotTrainedError
from ..log import count_passage

TokenBag = list[str]
IntermediateValue = t.TypeVar("IntermediateValue")
Features = dict[str, int]


log = logging.getLogger(__name__)


class VanillaSA(BaseSA):
    """
    A sentiment analyzer resembling the one implemented in structure the one implemented in the classroom, using the basic sentiment analyzer of NLTK.
    """

    def __init__(self, *, extractor: t.Callable[[Input], tuple[str, IntermediateValue]], tokenizer: t.Callable[[str], TokenBag], categorizer: t.Callable[[IntermediateValue], Category]) -> None:
        super().__init__()
        self.model: nltk.sentiment.SentimentAnalyzer = nltk.sentiment.SentimentAnalyzer()
        self.trained: bool = False
        self.extractor: t.Callable[[Input], tuple[str, IntermediateValue]] = extractor
        self.tokenizer: t.Callable[[str], TokenBag] = tokenizer
        self.categorizer: t.Callable[[IntermediateValue], Category] = categorizer

    def __repr__(self):
        return f"<{self.__class__.__qualname__} {'trained' if self.trained else 'untrained'} tokenizer={self.extractor!r} categorizer={self.categorizer!r}>"

    @staticmethod
    def __data_to_tokenbag(data: tuple[TokenBag, Category]) -> TokenBag:
        """
        Access the tokenbag of a data tuple.
        """
        return data[0]

    def __add_feature_unigrams(self, dataset: t.Iterator[tuple[TokenBag, Category]]) -> None:
        """
        Register the `nltk.sentiment.util.extract_unigram_feats` feature extrator on the model.
        """
        tokenbags = map(self.__data_to_tokenbag, dataset)
        all_words = self.model.all_words(tokenbags, labeled=False)
        unigrams = self.model.unigram_word_feats(words=all_words, min_freq=4)
        self.model.add_feat_extractor(nltk.sentiment.util.extract_unigram_feats, unigrams=unigrams)

    def _add_features(self, dataset: t.Iterator[tuple[TokenBag, Category]]):
        """
        Register new feature extractors on the `.model`.
        """
        self.__add_feature_unigrams(dataset)

    def __extract_features(self, data: tuple[TokenBag, Category]) -> tuple[Features, Category]:
        """
        Convert a (TokenBag, Category) tuple to a (Features, Category) tuple.

        Does not use `SentimentAnalyzer.apply_features` due to unexpected behaviour when using iterators.
        """
        count_passage("processed_features", 100)
        return self.model.extract_features(data[0]), data[1]

    def _train_from_dataset(self, dataset: t.Iterator[tuple[TokenBag, Category]]) -> None:
        """
        Train the model with the given training set.
        """
        if self.trained:
            raise AlreadyTrainedError()

        dataset_1, dataset_2 = itertools.tee(dataset, 2)

        self._add_features(dataset_1)
        del dataset_1

        dataset_2 = map(self.__extract_features, dataset_2)
        self.model.classifier = nltk.classify.NaiveBayesClassifier.train(dataset_2)
        self.trained = True

    def _evaluate_from_dataset(self, dataset: t.Iterator[tuple[TokenBag, Category]]) -> dict:
        """
        Perform a model evaluation with the given test set.
        """
        if not self.trained:
            raise NotTrainedError()

        dataset_1 = map(self.__extract_features, dataset)
        # FIXME: This won't work with streams :(
        return self.model.evaluate(list(dataset_1))

    def _use_from_tokenbag(self, tokens: TokenBag) -> Category:
        """
        Categorize the given token bag.
        """
        if not self.trained:
            raise NotTrainedError()

        return self.model.classify(instance=tokens)

    def _extract_data(self, inp: Input) -> tuple[TokenBag, Category]:
        count_passage("processed_data", 100)
        text, value = self.extractor(inp)
        return self.tokenizer(text), self.categorizer(value)

    def _extract_dataset(self, inp: t.Iterator[Input]) -> list[tuple[TokenBag, Category]]:
        return map(self._extract_data, inp)

    def train(self, training_set: t.Iterator[Input]) -> None:
        dataset = self._extract_dataset(training_set)
        self._train_from_dataset(dataset)

    def evaluate(self, test_set: t.Iterator[Input]) -> dict:
        dataset = self._extract_dataset(test_set)
        return self._evaluate_from_dataset(dataset)

    def use(self, text: Input) -> Category:
        tokens = self.tokenizer(text)
        return self._use_from_tokenbag(tokens)


__all__ = (
    "VanillaSA",
)

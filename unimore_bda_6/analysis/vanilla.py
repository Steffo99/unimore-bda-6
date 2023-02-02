import nltk
import nltk.classify
import nltk.sentiment
import nltk.sentiment.util
import logging
import typing as t

from .base import Input, Category, BaseSA, AlreadyTrainedError, NotTrainedError

TokenBag = list[str]
IntermediateValue = t.TypeVar("IntermediateValue")


log = logging.getLogger(__name__)


class VanillaSA(BaseSA):
    """
    A sentiment analyzer resembling the one implemented in structure the one implemented in the classroom, using the basic sentiment analyzer of NLTK.
    """

    def __init__(self, *, extractor: t.Callable[[Input], tuple[str, Category]], tokenizer: t.Callable[[str], TokenBag], categorizer: t.Callable[[Input], Category]) -> None:
        super().__init__()
        self.model: nltk.sentiment.SentimentAnalyzer = nltk.sentiment.SentimentAnalyzer()
        self.trained: bool = False

        self.extractor: t.Callable[[Input], tuple[str, IntermediateValue]] = extractor
        self.tokenizer: t.Callable[[str], TokenBag] = tokenizer
        self.categorizer: t.Callable[[IntermediateValue], Category] = categorizer

    def __add_feature_unigrams(self, training_set: list[tuple[TokenBag, Category]]) -> None:
        """
        Add the `nltk.sentiment.util.extract_unigram_feats` feature to the model.
        """
        all_words = self.model.all_words(training_set, labeled=True)
        unigrams = self.model.unigram_word_feats(words=all_words, min_freq=4)
        self.model.add_feat_extractor(nltk.sentiment.util.extract_unigram_feats, unigrams=unigrams)

    def _add_features(self, training_set: list[tuple[TokenBag, Category]]):
        """
        Add new features to the sentiment analyzer.
        """
        self.__add_feature_unigrams(training_set)

    def _train_from_dataset(self, dataset: list[tuple[TokenBag, Category]]) -> None:
        """
        Train the model with the given training set.
        """
        if self.trained:
            raise AlreadyTrainedError()

        self.__add_feature_unigrams(dataset)
        training_set_with_features = self.model.apply_features(dataset, labeled=True)

        self.model.train(trainer=nltk.classify.NaiveBayesClassifier.train, training_set=training_set_with_features)
        self.trained = True

    def _evaluate_from_dataset(self, dataset: list[tuple[TokenBag, Category]]) -> dict:
        """
        Perform a model evaluation with the given test set.
        """
        if not self.trained:
            raise NotTrainedError()

        test_set_with_features = self.model.apply_features(dataset, labeled=True)
        return self.model.evaluate(test_set_with_features)

    def _use_from_tokenbag(self, tokens: TokenBag) -> Category:
        """
        Categorize the given token bag.
        """
        if not self.trained:
            raise NotTrainedError()

        return self.model.classify(instance=tokens)

    def _extract_data(self, inp: Input) -> tuple[TokenBag, Category]:
        text, value = self.extractor(inp)
        return self.tokenizer(text), self.categorizer(value)

    def _extract_dataset(self, inp: list[Input]) -> list[tuple[TokenBag, Category]]:
        return list(map(self._extract_data, inp))

    def train(self, training_set: list[Input]) -> None:
        dataset = self._extract_dataset(training_set)
        self._train_from_dataset(dataset)

    def evaluate(self, test_set: list[tuple[Input, Category]]) -> None:
        dataset = self._extract_dataset(test_set)
        return self._evaluate_from_dataset(dataset)

    def use(self, text: Input) -> Category:
        tokens = self.tokenizer(text)
        return self._use_from_tokenbag(tokens)


__all__ = (
    "VanillaSA",
)

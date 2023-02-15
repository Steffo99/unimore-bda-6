import abc
import typing as t

import numpy
import tensorflow
import logging

from ..database import CachedDatasetFunc, TextReview, TokenizedReview
from ..config import TENSORFLOW_EMBEDDING_SIZE, TENSORFLOW_MAX_FEATURES, TENSORFLOW_EPOCHS
from ..tokenizer import BaseTokenizer
from .base import BaseSentimentAnalyzer, AlreadyTrainedError, NotTrainedError, TrainingFailedError

log = logging.getLogger(__name__)


if len(tensorflow.config.list_physical_devices(device_type="GPU")) == 0:
    log.warning("Tensorflow reports no GPU acceleration available.")
else:
    log.debug("Tensorflow successfully found GPU acceleration!")


ConversionFunc = t.Callable[[TextReview], tensorflow.Tensor | tuple]


class TensorflowSentimentAnalyzer(BaseSentimentAnalyzer, metaclass=abc.ABCMeta):
    """
    Base class for a sentiment analyzer using `tensorflow`.
    """

    def __init__(self, *, tokenizer: BaseTokenizer):
        super().__init__(tokenizer=tokenizer)

        self.trained: bool = False
        self.failed: bool = False

        self.string_lookup_layer: tensorflow.keras.layers.StringLookup = tensorflow.keras.layers.StringLookup(max_tokens=TENSORFLOW_MAX_FEATURES.__wrapped__)
        self.model: tensorflow.keras.Sequential = self._build_model()
        self.history: tensorflow.keras.callbacks.History | None = None

    @abc.abstractmethod
    def _build_model(self) -> tensorflow.keras.Sequential:
        """
        Create the `tensorflow.keras.Sequential` model that should be executed by this sentiment analyzer.
        """
        raise NotImplementedError()

    def _build_dataset(self, dataset_func: CachedDatasetFunc) -> tensorflow.data.Dataset:
        """
        Create a `tensorflow.data.Dataset` from the given `CachedDatasetFunc`.
        """

        def dataset_generator():
            for review in dataset_func():
                review: TextReview
                review: TokenizedReview = self.tokenizer.tokenize_review(review)
                tokens: tensorflow.Tensor = self._tokens_to_tensor(review.tokens)
                rating: tensorflow.Tensor = self._rating_to_input(review.rating)
                yield tokens, rating

        log.debug("Creating dataset...")
        dataset = tensorflow.data.Dataset.from_generator(
            dataset_generator,
            output_signature=(
                tensorflow.TensorSpec(shape=(1, None,), dtype=tensorflow.string, name="tokens"),
                self._ratingtensor_shape(),
            ),
        )

        log.debug("Caching dataset...")
        dataset = dataset.cache()

        log.debug("Configuring dataset prefetch...")
        dataset = dataset.prefetch(buffer_size=tensorflow.data.AUTOTUNE)

        return dataset

    def train(self, training_dataset_func: CachedDatasetFunc, validation_dataset_func: CachedDatasetFunc) -> None:
        if self.failed:
            log.error("Tried to train a failed model.")
            raise AlreadyTrainedError("Cannot re-train a failed model.")
        if self.trained:
            log.error("Tried to train an already trained model.")
            raise AlreadyTrainedError("Cannot re-train an already trained model.")

        log.debug("Building training dataset...")
        training_set = self._build_dataset(training_dataset_func)

        log.debug("Building validation dataset...")
        validation_set = self._build_dataset(validation_dataset_func)

        log.debug("Building vocabulary...")
        vocabulary = training_set.map(lambda tokens, rating: tokens)

        log.debug("Adapting lookup layer to the vocabulary...")
        self.string_lookup_layer.adapt(vocabulary)

        log.info("Training: %s", self.model)
        self.history: tensorflow.keras.callbacks.History | None  = self.model.fit(
            training_set,
            validation_data=validation_set,
            epochs=TENSORFLOW_EPOCHS.__wrapped__,
            callbacks=[
                tensorflow.keras.callbacks.TerminateOnNaN()
            ],
        )

        if len(self.history.epoch) < TENSORFLOW_EPOCHS.__wrapped__:
            log.error("Model %s training failed: only %d epochs computed", self.model, len(self.history.epoch))
            self.failed = True
            raise TrainingFailedError()
        else:
            log.info("Model %s training succeeded!", self.model)
            self.trained = True

    @staticmethod
    def _tokens_to_tensor(tokens: t.Iterator[str]) -> tensorflow.Tensor:
        """
        Convert an iterator of tokens to a `tensorflow.Tensor`.
        """
        tensor = tensorflow.convert_to_tensor(
            [list(tokens)],
            dtype=tensorflow.string,
            name="tokens"
        )
        return tensor

    def use(self, text: str) -> float:
        if self.failed:
            raise NotTrainedError("Cannot use a failed model.")
        if not self.trained:
            raise NotTrainedError("Cannot use a non-trained model.")

        tokens = self.tokenizer.tokenize(text)
        tokens = self._tokens_to_tensor(tokens)
        prediction = self.model.predict(tokens, verbose=False)
        prediction = self._prediction_to_rating(prediction)
        return prediction

    @abc.abstractmethod
    def _rating_to_input(self, rating: float) -> tensorflow.Tensor:
        """
        Convert a review rating to a `tensorflow.Tensor`.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _ratingtensor_shape(self) -> tensorflow.TensorSpec:
        """
        Returns the shape of the tensor output by `._rating_to_tensor` and accepted as input by `._tensor_to_rating`.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _prediction_to_rating(self, prediction: tensorflow.Tensor) -> float:
        """
        Convert the results of `tensorflow.keras.Sequential.predict` into a review rating.
        """
        raise NotImplementedError()


class TensorflowCategorySentimentAnalyzer(TensorflowSentimentAnalyzer):
    """
    A `tensorflow`-based sentiment analyzer that considers each star rating as a separate category.
    """

    def _build_model(self) -> tensorflow.keras.Sequential:
        log.debug("Creating sequential categorizer model...")
        model = tensorflow.keras.Sequential([
            self.string_lookup_layer,
            tensorflow.keras.layers.Embedding(
                input_dim=TENSORFLOW_MAX_FEATURES.__wrapped__ + 1,
                output_dim=TENSORFLOW_EMBEDDING_SIZE.__wrapped__,
            ),
            tensorflow.keras.layers.Dropout(0.25),
            tensorflow.keras.layers.GlobalAveragePooling1D(),
            tensorflow.keras.layers.Dropout(0.25),
            tensorflow.keras.layers.Dense(8),
            tensorflow.keras.layers.Dropout(0.25),
            tensorflow.keras.layers.Dense(5, activation="softmax"),
        ])

        log.debug("Compiling model: %s", model)
        model.compile(
            optimizer=tensorflow.keras.optimizers.Adam(clipnorm=1.0),
            loss=tensorflow.keras.losses.CategoricalCrossentropy(),
            metrics=[
                tensorflow.keras.metrics.CategoricalAccuracy(),
            ]
        )

        log.debug("Compiled model: %s", model)
        return model

    def _rating_to_input(self, rating: float) -> tensorflow.Tensor:
        tensor = tensorflow.convert_to_tensor(
            [[
                1.0 if rating == 1.0 else 0.0,
                1.0 if rating == 2.0 else 0.0,
                1.0 if rating == 3.0 else 0.0,
                1.0 if rating == 4.0 else 0.0,
                1.0 if rating == 5.0 else 0.0,
            ]],
            dtype=tensorflow.float32,
            name="rating_one_hot"
        )
        return tensor

    def _ratingtensor_shape(self) -> tensorflow.TensorSpec:
        spec = tensorflow.TensorSpec(shape=(1, 5), dtype=tensorflow.float32, name="rating_one_hot")
        return spec

    def _prediction_to_rating(self, prediction: tensorflow.Tensor) -> float:
        best_prediction = None
        best_prediction_index = None

        for index, prediction in enumerate(iter(prediction[0])):
            if best_prediction is None or prediction > best_prediction:
                best_prediction = prediction
                best_prediction_index = index

        result = float(best_prediction_index) + 1.0
        return result


class TensorflowPolarSentimentAnalyzer(TensorflowSentimentAnalyzer):
    """
    A `tensorflow`-based sentiment analyzer that uses the floating point value rating to get as close as possible to the correct category.
    """

    def _build_model(self) -> tensorflow.keras.Sequential:
        log.debug("Creating sequential categorizer model...")
        model = tensorflow.keras.Sequential([
            self.string_lookup_layer,
            tensorflow.keras.layers.Embedding(
                input_dim=TENSORFLOW_MAX_FEATURES.__wrapped__ + 1,
                output_dim=TENSORFLOW_EMBEDDING_SIZE.__wrapped__,
            ),
            tensorflow.keras.layers.Dropout(0.25),
            tensorflow.keras.layers.GlobalAveragePooling1D(),
            tensorflow.keras.layers.Dropout(0.25),
            tensorflow.keras.layers.Dense(8),
            tensorflow.keras.layers.Dropout(0.25),
            tensorflow.keras.layers.Dense(1, activation=tensorflow.keras.activations.sigmoid),
        ])

        log.debug("Compiling model: %s", model)
        model.compile(
            optimizer=tensorflow.keras.optimizers.Adam(clipnorm=1.0),
            loss=tensorflow.keras.losses.MeanAbsoluteError(),
        )

        log.debug("Compiled model: %s", model)
        return model

    def _rating_to_input(self, rating: float) -> tensorflow.Tensor:
        normalized_rating = (rating - 1) / 4
        tensor = tensorflow.convert_to_tensor(
            [normalized_rating],
            dtype=tensorflow.float32,
            name="rating_value"
        )
        return tensor

    def _ratingtensor_shape(self) -> tensorflow.TensorSpec:
        spec = tensorflow.TensorSpec(shape=(1,), dtype=tensorflow.float32, name="rating_value")
        return spec

    def _prediction_to_rating(self, prediction: numpy.array) -> float:
        rating: float = prediction[0, 0]
        rating = 1.0 if rating < 0.5 else 5.0
        return rating


__all__ = (
    "TensorflowSentimentAnalyzer",
    "TensorflowCategorySentimentAnalyzer",
    "TensorflowPolarSentimentAnalyzer",
)

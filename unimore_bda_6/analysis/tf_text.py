import abc
import typing as t

import numpy
import tensorflow
import logging

from ..database import Text, Category, CachedDatasetFunc, Review
from ..config import TENSORFLOW_EMBEDDING_SIZE, TENSORFLOW_MAX_FEATURES, TENSORFLOW_EPOCHS
from ..tokenizer import BaseTokenizer
from .base import BaseSentimentAnalyzer, AlreadyTrainedError, NotTrainedError, TrainingFailedError

log = logging.getLogger(__name__)


if len(tensorflow.config.list_physical_devices(device_type="GPU")) == 0:
    log.warning("Tensorflow reports no GPU acceleration available.")
else:
    log.debug("Tensorflow successfully found GPU acceleration!")


ConversionFunc = t.Callable[[Review], tensorflow.Tensor | tuple]


def build_dataset(dataset_func: CachedDatasetFunc, conversion_func: ConversionFunc, output_signature: tensorflow.TensorSpec | tuple) -> tensorflow.data.Dataset:
    """
    Convert a `CachedDatasetFunc` to a `tensorflow.data.Dataset`.
    """

    def dataset_generator():
        for review in dataset_func():
            yield conversion_func(review)

    log.debug("Creating dataset...")
    dataset = tensorflow.data.Dataset.from_generator(
        dataset_generator,
        output_signature=output_signature,
    )

    log.debug("Caching dataset...")
    dataset = dataset.cache()

    log.debug("Configuring dataset prefetch...")
    dataset = dataset.prefetch(buffer_size=tensorflow.data.AUTOTUNE)

    return dataset


class TensorflowSentimentAnalyzer(BaseSentimentAnalyzer, metaclass=abc.ABCMeta):
    """
    Base class for a sentiment analyzer using `tensorflow`.
    """

    def __init__(self, *, tokenizer: BaseTokenizer):
        if not tokenizer.supports_tensorflow():
            raise TypeError("Tokenizer does not support Tensorflow")

        super().__init__(tokenizer=tokenizer)

        self.trained: bool = False
        self.failed: bool = False

        self.tokenizer: BaseTokenizer = tokenizer
        self.text_vectorization_layer: tensorflow.keras.layers.TextVectorization = self._build_text_vectorization_layer()
        self.model: tensorflow.keras.Sequential = self._build_model()
        self.history: tensorflow.keras.callbacks.History | None = None

    def _build_text_vectorization_layer(self) -> tensorflow.keras.layers.TextVectorization:
        """
        Create a `tensorflow`-compatible `TextVectorization` layer.
        """
        log.debug("Creating TextVectorization layer...")
        layer = tensorflow.keras.layers.TextVectorization(
            standardize=self.tokenizer.tokenize_tensorflow_and_expand_dims,
            max_tokens=TENSORFLOW_MAX_FEATURES.__wrapped__
        )
        log.debug("Created TextVectorization layer: %s", layer)
        return layer

    @abc.abstractmethod
    def _build_model(self) -> tensorflow.keras.Sequential:
        """
        Create the `tensorflow.keras.Sequential` model that should be executed by this sentiment analyzer.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _build_dataset(self, dataset_func: CachedDatasetFunc) -> tensorflow.data.Dataset:
        """
        Create a `tensorflow.data.Dataset` from the given `CachedDatasetFunc`.
        """
        raise NotImplementedError()

    def _adapt_textvectorization(self, dataset: tensorflow.data.Dataset) -> None:
        """
        Adapt the `.text_vectorization_layer` to the given dataset.
        """
        log.debug("Preparing dataset to adapt %s...", self.text_vectorization_layer)
        dataset = dataset.map(lambda text, category: text)
        log.debug("Adapting %s...", self.text_vectorization_layer)
        self.text_vectorization_layer.adapt(dataset)

    def _vectorize_dataset(self, dataset: tensorflow.data.Dataset) -> tensorflow.data.Dataset:
        """
        Apply the `.text_vectorization_layer` to the text in the dataset.
        """
        def vectorize_entry(text, category):
            return self.text_vectorization_layer(text), category

        log.debug("Vectorizing dataset: %s", dataset)
        dataset = dataset.map(vectorize_entry)
        log.debug("Vectorized dataset: %s", dataset)
        return dataset

    def train(self, training_dataset_func: CachedDatasetFunc, validation_dataset_func: CachedDatasetFunc) -> None:
        if self.failed:
            log.error("Tried to train a failed model.")
            raise AlreadyTrainedError("Cannot re-train a failed model.")
        if self.trained:
            log.error("Tried to train an already trained model.")
            raise AlreadyTrainedError("Cannot re-train an already trained model.")

        training_set = self._build_dataset(training_dataset_func)
        validation_set = self._build_dataset(validation_dataset_func)

        self._adapt_textvectorization(training_set)

        training_set = self._vectorize_dataset(training_set)
        validation_set = self._vectorize_dataset(validation_set)

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

    @abc.abstractmethod
    def _translate_prediction(self, a: numpy.array) -> Category:
        """
        Convert the results of `tensorflow.keras.Sequential.predict` into a `.Category`.
        """
        raise NotImplementedError()

    def use(self, text: Text) -> Category:
        if self.failed:
            log.error("Tried to use a failed model.")
            raise NotTrainedError("Cannot use a failed model.")
        if not self.trained:
            log.error("Tried to use a non-trained model.")
            raise NotTrainedError("Cannot use a non-trained model.")

        vector = self.text_vectorization_layer(text)
        prediction = self.model.predict(vector, verbose=False)

        return self._translate_prediction(prediction)


class TensorflowCategorySentimentAnalyzer(TensorflowSentimentAnalyzer):
    """
    A `tensorflow`-based sentiment analyzer that considers each star rating as a separate category.
    """

    def _build_dataset(self, dataset_func: CachedDatasetFunc) -> tensorflow.data.Dataset:
        return build_dataset(
            dataset_func=dataset_func,
            conversion_func=Review.to_tensor_tuple_category,
            output_signature=(
                tensorflow.TensorSpec(shape=(), dtype=tensorflow.string, name="text"),
                tensorflow.TensorSpec(shape=(1, 5,), dtype=tensorflow.float32, name="category_one_hot"),
            ),
        )

    def _build_model(self) -> tensorflow.keras.Sequential:
        log.debug("Creating sequential categorizer model...")
        model = tensorflow.keras.Sequential([
            tensorflow.keras.layers.Embedding(
                input_dim=TENSORFLOW_MAX_FEATURES.__wrapped__ + 1,
                output_dim=TENSORFLOW_EMBEDDING_SIZE.__wrapped__,
            ),
            tensorflow.keras.layers.Dropout(0.25),
            tensorflow.keras.layers.GlobalAveragePooling1D(),
            tensorflow.keras.layers.Dropout(0.25),
            tensorflow.keras.layers.Dense(5, activation="softmax"),
        ])

        log.debug("Compiling model: %s", model)
        model.compile(
            optimizer=tensorflow.keras.optimizers.Adam(global_clipnorm=1.0),
            loss=tensorflow.keras.losses.CategoricalCrossentropy(),
            metrics=[
                tensorflow.keras.metrics.CategoricalAccuracy(),
            ]
        )

        log.debug("Compiled model: %s", model)
        return model

    def _translate_prediction(self, a: numpy.array) -> Category:
        max_i = None
        max_p = None
        for i, p in enumerate(iter(a[0])):
            if max_p is None or p > max_p:
                max_i = i
                max_p = p
        result = float(max_i) + 1.0
        return result


class TensorflowPolarSentimentAnalyzer(TensorflowSentimentAnalyzer):
    """
    A `tensorflow`-based sentiment analyzer that uses the floating point value rating to get as close as possible to the correct category.
    """

    def _build_dataset(self, dataset_func: CachedDatasetFunc) -> tensorflow.data.Dataset:
        return build_dataset(
            dataset_func=dataset_func,
            conversion_func=Review.to_tensor_tuple_normvalue,
            output_signature=(
                tensorflow.TensorSpec(shape=(), dtype=tensorflow.string, name="text"),
                tensorflow.TensorSpec(shape=(1,), dtype=tensorflow.float32, name="category"),
            ),
        )

    def _build_model(self) -> tensorflow.keras.Sequential:
        log.debug("Creating sequential categorizer model...")
        model = tensorflow.keras.Sequential([
            tensorflow.keras.layers.Embedding(
                input_dim=TENSORFLOW_MAX_FEATURES.__wrapped__ + 1,
                output_dim=TENSORFLOW_EMBEDDING_SIZE.__wrapped__,
            ),
            tensorflow.keras.layers.Dropout(0.25),
            tensorflow.keras.layers.GlobalAveragePooling1D(),
            tensorflow.keras.layers.Dropout(0.25),
            tensorflow.keras.layers.Dense(1),
        ])

        log.debug("Compiling model: %s", model)
        model.compile(
            optimizer=tensorflow.keras.optimizers.Adam(global_clipnorm=1.0),
            loss=tensorflow.keras.losses.MeanSquaredError(),
            metrics=[
                tensorflow.keras.metrics.MeanAbsoluteError(),
                tensorflow.keras.metrics.CosineSimilarity(),
            ]
        )

        log.debug("Compiled model: %s", model)
        return model

    def _translate_prediction(self, a: numpy.array) -> Category:
        return a[0, 0] * 5


__all__ = (
    "TensorflowSentimentAnalyzer",
    "TensorflowCategorySentimentAnalyzer",
    "TensorflowPolarSentimentAnalyzer",
)

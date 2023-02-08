import tensorflow
import logging

from ..database import Text, Category, CachedDatasetFunc
from ..config import TENSORFLOW_EMBEDDING_SIZE, TENSORFLOW_MAX_FEATURES, TENSORFLOW_EPOCHS
from ..tokenizer import BaseTokenizer
from .base import BaseSentimentAnalyzer, AlreadyTrainedError, NotTrainedError, TrainingFailedError

log = logging.getLogger(__name__)


if len(tensorflow.config.list_physical_devices(device_type="GPU")) == 0:
    log.warning("Tensorflow reports no GPU acceleration available.")
else:
    log.debug("Tensorflow successfully found GPU acceleration!")


class TensorflowSentimentAnalyzer(BaseSentimentAnalyzer):
    def __init__(self, *, tokenizer: BaseTokenizer):
        if not tokenizer.supports_tensorflow():
            raise TypeError("Tokenizer does not support Tensorflow")

        super().__init__(tokenizer=tokenizer)

        self.trained: bool = False

        self.text_vectorization_layer: tensorflow.keras.layers.TextVectorization = self._build_vectorizer(tokenizer)
        self.model: tensorflow.keras.Sequential = self._build_model()
        self.history: tensorflow.keras.callbacks.History | None = None

    @staticmethod
    def _build_dataset(dataset_func: CachedDatasetFunc) -> tensorflow.data.Dataset:
        """
        Convert a `CachedDatasetFunc` to a `tensorflow.data.Dataset`.
        """

        def dataset_func_with_tensor_tuple():
            for review in dataset_func():
                yield review.to_tensor_tuple()

        log.debug("Creating dataset...")
        dataset = tensorflow.data.Dataset.from_generator(
            dataset_func_with_tensor_tuple,
            output_signature=(
                tensorflow.TensorSpec(shape=(), dtype=tensorflow.string, name="text"),
                tensorflow.TensorSpec(shape=(1, 5,), dtype=tensorflow.float32, name="category"),
            )
        )

        log.debug("Caching dataset...")
        dataset = dataset.cache()

        log.debug("Configuring dataset prefetch...")
        dataset = dataset.prefetch(buffer_size=tensorflow.data.AUTOTUNE)

        return dataset

    @staticmethod
    def _build_model() -> tensorflow.keras.Sequential:
        log.debug("Creating model...")
        model = tensorflow.keras.Sequential([
            tensorflow.keras.layers.Embedding(
                input_dim=TENSORFLOW_MAX_FEATURES.__wrapped__ + 1,
                output_dim=TENSORFLOW_EMBEDDING_SIZE.__wrapped__,
            ),
            tensorflow.keras.layers.Dropout(0.25),
            tensorflow.keras.layers.GlobalAveragePooling1D(),
            tensorflow.keras.layers.Dropout(0.25),
            tensorflow.keras.layers.Dense(25),
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

    @staticmethod
    def _build_vectorizer(tokenizer: BaseTokenizer) -> tensorflow.keras.layers.TextVectorization:
        return tensorflow.keras.layers.TextVectorization(
            standardize=tokenizer.tokenize_tensorflow,
            max_tokens=TENSORFLOW_MAX_FEATURES.__wrapped__
        )

    def train(self, training_dataset_func: CachedDatasetFunc, validation_dataset_func: CachedDatasetFunc) -> None:
        if self.trained:
            log.error("Tried to train an already trained model.")
            raise AlreadyTrainedError()

        log.debug("Building datasets...")
        training_set = self._build_dataset(training_dataset_func)
        validation_set = self._build_dataset(validation_dataset_func)
        log.debug("Built dataset: %s", training_set)

        log.debug("Preparing training_set for %s...", self.text_vectorization_layer.adapt)
        only_text_set = training_set.map(lambda text, category: text)

        log.debug("Adapting text_vectorization_layer: %s", self.text_vectorization_layer)
        self.text_vectorization_layer.adapt(only_text_set)
        log.debug("Adapted text_vectorization_layer: %s", self.text_vectorization_layer)

        log.debug("Preparing training_set for %s...", self.model.fit)
        training_set = training_set.map(lambda text, category: (self.text_vectorization_layer(text), category))
        validation_set = validation_set.map(lambda text, category: (self.text_vectorization_layer(text), category))
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
            raise TrainingFailedError()
        else:
            log.info("Model %s training succeeded!", self.model)

        self.trained = True

    def use(self, text: Text) -> Category:
        if not self.trained:
            log.error("Tried to use a non-trained model.")
            raise NotTrainedError()

        vector = self.text_vectorization_layer(text)

        prediction = self.model.predict(vector, verbose=False)

        max_i = None
        max_p = None
        for i, p in enumerate(iter(prediction[0])):
            if max_p is None or p > max_p:
                max_i = i
                max_p = p
        result = float(max_i) + 1.0

        return result

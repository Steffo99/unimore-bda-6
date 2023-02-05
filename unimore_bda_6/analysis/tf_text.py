import tensorflow
import itertools
import typing as t

from ..database import Text, Category, Review, DatasetFunc
from ..tokenizer import BaseTokenizer
from .base import BaseSentimentAnalyzer, AlreadyTrainedError, NotTrainedError


class TensorflowSentimentAnalyzer(BaseSentimentAnalyzer):
    def __init__(self, *, tokenizer: BaseTokenizer):
        super().__init__()
        self.trained = False
        self.neural_network: tensorflow.keras.Sequential | None = None
        self.tokenizer: BaseTokenizer = tokenizer  # TODO

    MAX_FEATURES = 20000
    EMBEDDING_DIM = 16
    EPOCHS = 10

    def train(self, dataset_func: DatasetFunc) -> None:
        if self.trained:
            raise AlreadyTrainedError()

        def dataset_func_with_tensor_text():
            for review in dataset_func():
                yield review.to_tensor_text()

        text_set = tensorflow.data.Dataset.from_generator(
            dataset_func_with_tensor_text,
            output_signature=tensorflow.TensorSpec(shape=(), dtype=tensorflow.string)
        )

        text_vectorization_layer = tensorflow.keras.layers.TextVectorization(
            max_tokens=self.MAX_FEATURES,
            standardize=self.tokenizer.tokenize_tensorflow,
        )
        text_vectorization_layer.adapt(text_set)

        def dataset_func_with_tensor_tuple():
            for review in dataset_func():
                yield review.to_tensor_tuple()

        training_set = tensorflow.data.Dataset.from_generator(
            dataset_func_with_tensor_tuple,
            output_signature=(
                tensorflow.TensorSpec(shape=(), dtype=tensorflow.string, name="text"),
                tensorflow.TensorSpec(shape=(), dtype=tensorflow.float32, name="category"),
            )
        )

        # I have no idea of what I'm doing here
        self.neural_network = tensorflow.keras.Sequential([
            tensorflow.keras.layers.Embedding(self.MAX_FEATURES + 1, self.EMBEDDING_DIM),
            tensorflow.keras.layers.Dropout(0.2),
            tensorflow.keras.layers.GlobalAveragePooling1D(),
            tensorflow.keras.layers.Dropout(0.2),
            tensorflow.keras.layers.Dense(1),
        ])

        self.neural_network.compile(
            loss=tensorflow.losses.BinaryCrossentropy(from_logits=True),  # Only works with two tags
            metrics=tensorflow.metrics.BinaryAccuracy(threshold=0.0)
        )

        training_set = training_set.map(text_vectorization_layer)

        self.neural_network.fit(
            training_set,
            epochs=self.EPOCHS,
        )

        self.trained = True

    def use(self, text: Text) -> Category:
        if not self.trained:
            raise NotTrainedError()

        prediction = self.neural_network.predict(text)
        breakpoint()

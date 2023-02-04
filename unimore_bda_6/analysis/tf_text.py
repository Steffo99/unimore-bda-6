import tensorflow
import itertools
import typing as t

from ..database import DataSet, Text, Category
from ..tokenizer import BaseTokenizer
from .base import BaseSentimentAnalyzer, AlreadyTrainedError, NotTrainedError


class TensorflowSentimentAnalyzer(BaseSentimentAnalyzer):
    def __init__(self, *, tokenizer: BaseTokenizer):
        super().__init__(tokenizer=tokenizer)
        self.trained = False
        self.text_vectorization_layer = None
        self.neural_network: tensorflow.keras.Sequential | None = None

    @staticmethod
    def __infinite_dataset_generator_factory(dataset: DataSet):
        """
        A generator of infinite copies of dataset.

        .. todo:: Loads the whole dataset in memory. What a waste! Can we perform multiple MongoDB queries instead?
        """
        dataset = map(lambda text, category: (tensorflow.convert_to_tensor(text, dtype=tensorflow.string), tensorflow.convert_to_tensor(category, dtype=tensorflow.string)), dataset)

        def generator():
            while True:
                nonlocal dataset
                dataset, result = itertools.tee(dataset, 2)
                yield result

        return generator

    @classmethod
    def __bda_dataset_to_tf_dataset(cls, dataset: DataSet) -> tensorflow.data.Dataset:
        """
        Convert a `unimore_bda_6.database.DataSet` to a "real" `tensorflow.data.Dataset`.
        """
        return tensorflow.data.Dataset.from_generator(
            cls.__infinite_dataset_generator_factory(dataset),
            output_signature=(
                tensorflow.TensorSpec(shape=(), dtype=tensorflow.string),
                tensorflow.TensorSpec(shape=(), dtype=tensorflow.string),
            )
        )

    MAX_FEATURES = 20000
    EMBEDDING_DIM = 16
    EPOCHS = 10

    def train(self, training_set: DataSet) -> None:
        if self.trained:
            raise AlreadyTrainedError()

        training_set = self.__bda_dataset_to_tf_dataset(training_set)

        self.text_vectorization_layer = tensorflow.keras.layers.TextVectorization(
            max_tokens=self.MAX_FEATURES,
            standardize=self.tokenizer.tokenize_tensorflow,
        )
        self.text_vectorization_layer.adapt(map(lambda t: t[0], training_set))

        training_set = training_set.map(self.text_vectorization_layer)

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

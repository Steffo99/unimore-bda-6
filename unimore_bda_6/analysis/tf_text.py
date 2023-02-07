import tensorflow

from ..database import Text, Category, DatasetFunc
from ..config import DATA_SET_SIZE
from .base import BaseSentimentAnalyzer, AlreadyTrainedError, NotTrainedError


class TensorflowSentimentAnalyzer(BaseSentimentAnalyzer):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.trained: bool = False

        self.text_vectorization_layer: tensorflow.keras.layers.TextVectorization = self._build_vectorizer()
        self.model: tensorflow.keras.Sequential = self._build_model()

    def _build_dataset(self, dataset_func: DatasetFunc) -> tensorflow.data.Dataset:
        def dataset_func_with_tensor_tuple():
            for review in dataset_func():
                yield review.to_tensor_tuple()

        return tensorflow.data.Dataset.from_generator(
            dataset_func_with_tensor_tuple,
            output_signature=(
                tensorflow.TensorSpec(shape=(), dtype=tensorflow.string, name="text"),
                tensorflow.TensorSpec(shape=(5,), dtype=tensorflow.float32, name="category"),
            )
        )

    def _build_model(self) -> tensorflow.keras.Sequential:
        return tensorflow.keras.Sequential([
            tensorflow.keras.layers.Embedding(
                input_dim=self.MAX_FEATURES + 1,
                output_dim=self.EMBEDDING_DIM,
            ),
            # tensorflow.keras.layers.Dropout(0.2),
            tensorflow.keras.layers.GlobalAveragePooling1D(),
            # tensorflow.keras.layers.Dropout(0.2),
            tensorflow.keras.layers.Dense(5, activation="softmax"),
        ])

    def _build_vectorizer(self) -> tensorflow.keras.layers.TextVectorization:
        return tensorflow.keras.layers.TextVectorization(max_tokens=self.MAX_FEATURES)

    def __vectorize_data(self, text, category):
        text = tensorflow.expand_dims(text, -1)  # TODO: ??????
        return self.text_vectorization_layer(text), category

    MAX_FEATURES = 2500
    EMBEDDING_DIM = 24
    """
    Count of possible "semantic meanings" of words, represented as dimensions of a tensor.
    """

    EPOCHS = 3

    def train(self, dataset_func: DatasetFunc) -> None:
        if self.trained:
            raise AlreadyTrainedError()

        training_set = self._build_dataset(dataset_func)

        only_text_set = training_set.map(lambda text, category: text)
        self.text_vectorization_layer.adapt(only_text_set)
        training_set = training_set.map(self.__vectorize_data)

        # self.model.compile(loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer="adam", metrics=["accuracy"])
        self.model.compile(loss=tensorflow.keras.losses.MeanAbsoluteError(), optimizer="adam", metrics=["accuracy"])

        self.model.fit(training_set, epochs=self.EPOCHS)

        self.trained = True

    def use(self, text: Text) -> Category:
        if not self.trained:
            raise NotTrainedError()

        vector = self.text_vectorization_layer(tensorflow.expand_dims(text, -1))

        prediction = self.model.predict(vector)

        max_i = None
        max_p = None
        for i, p in enumerate(iter(prediction[0])):
            if max_p is None or p > max_p:
                max_i = i
                max_p = p

        return float(max_i) + 1.0

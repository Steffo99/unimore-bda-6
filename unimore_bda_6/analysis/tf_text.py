import tensorflow

from ..database import Text, Category, DatasetFunc
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
                tensorflow.TensorSpec(shape=(), dtype=tensorflow.float32, name="category"),
            )
        )

    def _build_model(self) -> tensorflow.keras.Sequential:
        return tensorflow.keras.Sequential([
            tensorflow.keras.layers.Embedding(input_dim=self.MAX_FEATURES + 1, output_dim=self.EMBEDDING_DIM),
            tensorflow.keras.layers.Dropout(0.2),
            tensorflow.keras.layers.GlobalAveragePooling1D(),
            tensorflow.keras.layers.Dropout(0.2),
            tensorflow.keras.layers.Dense(1),
        ])

    def _build_vectorizer(self) -> tensorflow.keras.layers.TextVectorization:
        return tensorflow.keras.layers.TextVectorization(max_tokens=self.MAX_FEATURES)

    def __vectorize_data(self, text, category):
        text = tensorflow.expand_dims(text, -1)  # TODO: ??????
        return self.text_vectorization_layer(text), category

    MAX_FEATURES = 1000
    EMBEDDING_DIM = 16
    EPOCHS = 10

    def train(self, dataset_func: DatasetFunc) -> None:
        if self.trained:
            raise AlreadyTrainedError()

        training_set = self._build_dataset(dataset_func)

        only_text_set = training_set.map(lambda text, category: text)
        self.text_vectorization_layer.adapt(only_text_set)
        training_set = training_set.map(self.__vectorize_data)

        self.model.compile(loss=tensorflow.keras.losses.CosineSimilarity(axis=0), metrics=["accuracy"])

        history = self.model.fit(training_set, epochs=self.EPOCHS)

        ...

        self.trained = True

    def use(self, text: Text) -> Category:
        if not self.trained:
            raise NotTrainedError()

        prediction = self.model.predict(text)
        breakpoint()

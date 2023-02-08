import tensorflow
from .collections import MongoReview
import logging

log = logging.getLogger(__name__)


Text = str
Category = float


class Review:
    def __init__(self, text: Text, category: Category):
        self.text: str = text
        self.category: float = category

    @classmethod
    def from_mongoreview(cls, review: MongoReview):
        return cls(
            text=review["reviewText"],
            category=review["overall"],
        )

    def __repr__(self):
        return f"<{self.__class__.__qualname__}: [{self.category}] {self.text}>"

    def __getitem__(self, item):
        if item == 0 or item == "text":
            return self.text
        elif item == 1 or item == "category":
            return self.category
        else:
            raise KeyError(item)

    def to_tensor_text(self) -> tensorflow.Tensor:
        return tensorflow.convert_to_tensor(self.text, dtype=tensorflow.string)

    def to_tensor_category(self) -> tensorflow.Tensor:
        return tensorflow.convert_to_tensor([[
            1.0 if self.category == 1.0 else 0.0,
            1.0 if self.category == 2.0 else 0.0,
            1.0 if self.category == 3.0 else 0.0,
            1.0 if self.category == 4.0 else 0.0,
            1.0 if self.category == 5.0 else 0.0,
        ]], dtype=tensorflow.float32)

    def to_tensor_tuple(self) -> tuple[tensorflow.Tensor, tensorflow.Tensor]:
        t = (
            self.to_tensor_text(),
            self.to_tensor_category(),
        )
        log.debug("Converted %s", t)
        return t


__all__ = (
    "Text",
    "Category",
    "Review",
)

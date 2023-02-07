import typing as t
import logging
import shutil
import pathlib
import pickle

from .datatypes import Review

log = logging.getLogger(__name__)


DatasetFunc = t.Callable[[], t.Generator[Review, t.Any, None]]


def store_cache(reviews: t.Iterator[Review], path: str | pathlib.Path) -> None:
    """
    Store the contents of the given `Review` iterator to different files in a directory at the given path.
    """
    path = pathlib.Path(path)

    if path.exists():
        raise FileExistsError("Specified cache path already exists.")

    # Create the temporary directory
    log.debug("Creating cache directory: %s", path)
    path.mkdir(parents=True)

    # Write the documents to path/{index}.pickle
    for index, document in enumerate(reviews):
        document_path = path.joinpath(f"{index}.pickle")

        log.debug("Storing pickle file: %s", document_path)
        with open(document_path, "wb") as file:
            pickle.dump(document, file)


def load_cache(path: str | pathlib.Path) -> DatasetFunc:
    """
    Load the contents of a directory
    """
    path = pathlib.Path(path)

    if not path.exists():
        raise FileNotFoundError("The specified path does not exist.")

    def data_cache_loader():
        document_paths = path.iterdir()
        for document_path in document_paths:
            document_path = pathlib.Path(document_path)
            if not str(document_path).endswith(".pickle"):
                log.debug("Ignoring non-pickle file: %s", document_path)

            log.debug("Loading pickle file: %s", document_path)
            with open(document_path, "rb") as file:
                result: Review = pickle.load(file)
                yield result

    return data_cache_loader


__all__ = (
    "DatasetFunc",
    "store_cache",
    "load_cache",
)

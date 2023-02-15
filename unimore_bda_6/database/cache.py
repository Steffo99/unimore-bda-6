import typing as t
import logging
import shutil
import pathlib
import pickle

from .datatypes import TextReview

log = logging.getLogger(__name__)


CachedDatasetFunc = t.Callable[[], t.Generator[TextReview, t.Any, None]]


def store_cache(reviews: t.Iterator[TextReview], path: str | pathlib.Path) -> None:
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


def load_cache(path: str | pathlib.Path) -> CachedDatasetFunc:
    """
    Load the contents of a directory into a `Review` generator.
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
                continue

            log.debug("Loading pickle file: %s", document_path)
            with open(document_path, "rb") as file:
                result: TextReview = pickle.load(file)
                yield result

    return data_cache_loader


def delete_cache(path: str | pathlib.Path) -> None:
    """
    Delete the given cache directory.
    """
    path = pathlib.Path(path)

    if not path.exists():
        raise FileNotFoundError("The specified path does not exist.")

    log.debug("Deleting cache directory: %s", path)
    shutil.rmtree(path)


__all__ = (
    "CachedDatasetFunc",
    "store_cache",
    "load_cache",
    "delete_cache",
)

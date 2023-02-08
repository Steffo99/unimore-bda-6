import pymongo
import pymongo.errors
import contextlib
import typing as t
import logging

from ..config import MONGO_HOST, MONGO_PORT

log = logging.getLogger(__name__)


@contextlib.contextmanager
def mongo_client_from_config() -> t.ContextManager[pymongo.MongoClient]:
    """
    Create a new MongoDB client and yield it.
    """
    log.debug("Creating MongoDB client...")
    client: pymongo.MongoClient = pymongo.MongoClient(
        host=MONGO_HOST.__wrapped__,
        port=MONGO_PORT.__wrapped__,
    )
    log.debug("Created MongoDB client!")

    yield client

    log.info("Closing connection to MongoDB...")
    client.close()
    log.debug("Closed connection to MongoDB!")


__all__ = (
    "mongo_client_from_config",
)

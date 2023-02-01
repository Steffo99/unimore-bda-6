import cfig

config = cfig.Configuration()


@config.required()
def MONGO_HOST(val: str) -> str:
    """
    The hostname of the MongoDB database to connect to.
    """
    return val


@config.required()
def MONGO_PORT(val: str) -> str:
    """
    The port of the MongoDB database to connect to.
    """
    return val


@config.required()
def TRAINING_SET_SIZE(val: str) -> int:
    """
    The number of reviews from each category to fetch for the training set.
    """
    try:
        return int(val)
    except ValueError:
        raise cfig.InvalidValueError("Not an int.")


__all__ = (
    "config",
    "MONGO_HOST",
    "MONGO_PORT",
)

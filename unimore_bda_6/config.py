import cfig

config = cfig.Configuration()


@config.optional()
def MONGO_HOST(val: str | None) -> str:
    """
    The hostname of the MongoDB database to connect to.
    """
    return val or "127.0.0.1"


@config.optional()
def MONGO_PORT(val: str | None) -> int:
    """
    The port of the MongoDB database to connect to.
    """
    if val is None:
        return 27017
    try:
        return int(val)
    except ValueError:
        raise cfig.InvalidValueError("Not an int.")


@config.optional()
def SAMPLE_MODE(val: str | None) -> str:
    """
    Whether `$sample` or `$limit` should be used to aggregate the training and test sets.
    `$limit` is much faster, but not truly random, while `$sample` is completely random.
    """
    if val is None:
        return "$sample"
    if val not in ["$sample", "$limit"]:
        raise cfig.InvalidValueError("Neither $sample or $limit.")
    return val


@config.optional()
def TRAINING_SET_SIZE(val: str | None) -> int:
    """
    The number of reviews from each category to fetch for the training set.
    """
    if val is None:
        return 1000
    try:
        return int(val)
    except ValueError:
        raise cfig.InvalidValueError("Not an int.")


@config.optional()
def TEST_SET_SIZE(val: str | None) -> int:
    """
    The number of reviews to fetch for the test set.
    """
    if val is None:
        return 1000
    try:
        return int(val)
    except ValueError:
        raise cfig.InvalidValueError("Not an int.")


__all__ = (
    "config",
    "MONGO_HOST",
    "MONGO_PORT",
    "SAMPLE_MODE",
    "TRAINING_SET_SIZE",
    "TEST_SET_SIZE",
)

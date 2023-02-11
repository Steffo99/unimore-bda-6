import cfig

config = cfig.Configuration()


@config.optional()
def MONGO_HOST(val: str | None) -> str:
    """
    The hostname of the MongoDB database to connect to.

    Defaults to `"127.0.0.1"`.
    """
    return val or "127.0.0.1"


@config.optional()
def MONGO_PORT(val: str | None) -> int:
    """
    The port of the MongoDB database to connect to.

    Defaults to `27017`.
    """
    if val is None:
        return 27017
    try:
        return int(val)
    except ValueError:
        raise cfig.InvalidValueError("Not an int.")


@config.optional()
def WORKING_SET_SIZE(val: str | None) -> int:
    """
    The number of reviews to consider from the database.
    Set this to a low number to prevent slowness due to the dataset's huge size.

    Defaults to `10000`.
    """
    if val is None:
        return 10000
    try:
        return int(val)
    except ValueError:
        raise cfig.InvalidValueError("Not an int.")


@config.optional()
def TRAINING_SET_SIZE(val: str | None) -> int:
    """
    The number of reviews from each category to fetch for the training dataset.

    Defaults to `5000`.
    """
    if val is None:
        return 5000
    try:
        return int(val)
    except ValueError:
        raise cfig.InvalidValueError("Not an int.")


@config.optional()
def VALIDATION_SET_SIZE(val: str | None) -> int:
    """
    The number of reviews from each category to fetch for the training dataset.

    Defaults to `400`.
    """
    if val is None:
        return 400
    try:
        return int(val)
    except ValueError:
        raise cfig.InvalidValueError("Not an int.")


@config.optional()
def EVALUATION_SET_SIZE(val: str | None) -> int:
    """
    The number of reviews from each category to fetch for the evaluation dataset.

    Defaults to `1000`.
    """
    if val is None:
        return 1000
    try:
        return int(val)
    except ValueError:
        raise cfig.InvalidValueError("Not an int.")


@config.optional()
def TENSORFLOW_MAX_FEATURES(val: str | None) -> int:
    """
    The maximum number of features to use in Tensorflow models.

    Defaults to `30000`.
    """
    if val is None:
        return 30000
    try:
        return int(val)
    except ValueError:
        raise cfig.InvalidValueError("Not an int.")


@config.optional()
def TENSORFLOW_EMBEDDING_SIZE(val: str | None) -> int:
    """
    The size of the embeddings tensor to use in Tensorflow models.

    Defaults to `12`.
    """
    if val is None:
        return 12
    try:
        return int(val)
    except ValueError:
        raise cfig.InvalidValueError("Not an int.")


@config.optional()
def TENSORFLOW_EPOCHS(val: str | None) -> int:
    """
    The number of epochs to train Tensorflow models for.

    Defaults to `5`.
    """
    if val is None:
        return 5
    try:
        return int(val)
    except ValueError:
        raise cfig.InvalidValueError("Not an int.")


__all__ = (
    "config",
    "MONGO_HOST",
    "MONGO_PORT",
    "WORKING_SET_SIZE",
    "TRAINING_SET_SIZE",
    "VALIDATION_SET_SIZE",
    "EVALUATION_SET_SIZE",
    "TENSORFLOW_MAX_FEATURES",
    "TENSORFLOW_EMBEDDING_SIZE",
    "TENSORFLOW_EPOCHS",
)


if __name__ == "__main__":
    config.cli()

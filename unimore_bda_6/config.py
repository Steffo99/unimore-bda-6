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
def DATA_SET_SIZE(val: str | None) -> int:
    """
    The number of reviews from each category to fetch for the datasets.

    Defaults to `1000`.
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
    "WORKING_SET_SIZE",
    "DATA_SET_SIZE",
)


if __name__ == "__main__":
    config.cli()

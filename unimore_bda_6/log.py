import collections
import logging
import coloredlogs
import pathlib

logging.addLevelName(11, "SUCCESS")
logging.addLevelName(12, "FAILURE")

this_log = logging.getLogger(__name__)


def install_general_log_handlers():
    """
    Setup the `unimore_bda_6` and ``__main__`` loggers.

    With colors!
    """
    main_logger: logging.Logger = logging.getLogger("__main__")

    interesting_loggers: list[logging.Logger] = [
        main_logger,
        logging.getLogger("unimore_bda_6"),
    ]

    chatty_loggers: list[logging.Logger] = [
        logging.getLogger("unimore_bda_6.database.cache"),
        logging.getLogger("unimore_bda_6.database.datatypes"),
    ]

    this_log.debug("Installing console handlers...")
    for logger in interesting_loggers:
        coloredlogs.install(
            logger=logger,
            level="DEBUG" if __debug__ else "INFO",
            fmt="{asctime} | {name} | {levelname} | {message}",
            style="{",
            level_styles=dict(
                debug=dict(color="white"),
                info=dict(color="cyan"),
                warning=dict(color="yellow", bold=True),
                error=dict(color="red", bold=True),
                critical=dict(color="black", background="red", bold=True),
                success=dict(color="green"),
                failure=dict(color="yellow"),
            ),
            field_styles=dict(
                asctime=dict(color='magenta'),
                levelname=dict(color='blue', bold=True),
                name=dict(color='blue'),
            ),
            isatty=True,
        )
        this_log.debug("Installed console log handler on: %s", logger)

    this_log.debug("Silencing chatty loggers...")
    for logger in chatty_loggers:
        logger.setLevel("INFO")
        this_log.debug("Silenced: %s", logger)

    log_file_path = pathlib.Path("./data/logs/run.tsv")
    log_directory_path = log_file_path.parent
    this_log.debug("Ensuring %s exists...", log_directory_path)
    log_directory_path.mkdir(parents=True, exist_ok=True)
    this_log.debug("Ensuring %s exists...", log_file_path)
    open(log_file_path, "w").close()
    this_log.debug("Installing FileHandler for the __main__ logger at %s...", log_file_path)
    file_handler = logging.FileHandler(log_file_path, mode="a")
    file_handler.formatter = logging.Formatter("{asctime}\t{name}\t{levelname}\t{message}", style="{")
    main_logger.addHandler(file_handler)
    this_log.debug("Installed FileHandler for the __main__ logger at %s!", log_file_path)


_passage_counts = collections.defaultdict(lambda: 0)


def count_passage(log: logging.Logger, key: str, mod: int):
    _passage_counts[key] += 1
    if not _passage_counts[key] % mod:
        log.debug("%s - %d calls", key, _passage_counts[key])


__all__ = (
    "install_general_log_handlers",
    "count_passage",
)

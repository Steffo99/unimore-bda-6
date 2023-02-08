import collections
import logging
import coloredlogs

this_log = logging.getLogger(__name__)


def install_log_handler(loggers: list[logging.Logger] = None):
    if loggers is None:
        loggers = [
            logging.getLogger("__main__"),
            logging.getLogger("unimore_bda_6"),
        ]

    for logger in loggers:
        coloredlogs.install(
            logger=logger,
            level="DEBUG" if __debug__ else "INFO",
            fmt="{asctime} | {name:<80} | {levelname:>8} | {message}",
            style="{",
            level_styles=dict(
                debug=dict(color="white"),
                info=dict(color="cyan"),
                warning=dict(color="yellow"),
                error=dict(color="red"),
                critical=dict(color="black", background="red", bold=True),
            ),
            field_styles=dict(
                asctime=dict(color='magenta'),
                levelname=dict(color='blue', bold=True),
                name=dict(color='blue'),
            ),
            isatty=True,
        )
        this_log.debug("Installed custom log handler on: %s", logger)

    logging.getLogger("unimore_bda_6.database.cache").setLevel("INFO")
    logging.getLogger("unimore_bda_6.database.datatypes").setLevel("INFO")


_passage_counts = collections.defaultdict(lambda: 0)


def count_passage(log: logging.Logger, key: str, mod: int):
    _passage_counts[key] += 1
    if not _passage_counts[key] % mod:
        log.debug("%s - %d calls", key, _passage_counts[key])


__all__ = (
    "install_log_handler",
    "count_passage",
)

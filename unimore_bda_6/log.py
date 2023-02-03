import collections
import logging
import coloredlogs

log = logging.getLogger(__name__)


def install_log_handler(loggers: list[logging.Logger] = None):
    if loggers is None:
        loggers = [
            logging.getLogger("__main__"),
            logging.getLogger("unimore_bda_6"),
        ]

    for logger in loggers:
        coloredlogs.install(
            logger=logger,
            level="DEBUG",
            fmt="{asctime} | {name:<32} | {levelname:>8} | {message}",
            style="{",
            level_styles=dict(
                debug=dict(color="white"),
                info=dict(color="cyan"),
                warning=dict(color="yellow"),
                error=dict(color="red"),
                critical=dict(color="red", bold=True),
            ),
            field_styles=dict(
                asctime=dict(color='magenta'),
                levelname=dict(color='blue', bold=True),
                name=dict(color='blue'),
            ),
            isatty=True,
        )
        log.debug("Installed custom log handler on: %s", logger)


_passage_counts = collections.defaultdict(lambda: 0)


def count_passage(key: str, mod: int):
    _passage_counts[key] += 1
    if not _passage_counts[key] % mod:
        log.debug("%s - %d calls", key, _passage_counts[key])


__all__ = (
    "install_log_handler",
    "count_passage",
)

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


__all__ = (
    "install_log_handler",
)

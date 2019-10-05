import logging, sys
import os
from structlog import wrap_logger
from structlog.processors import JSONRenderer, TimeStamper

from logging.handlers import RotatingFileHandler
from blackopt.config import get_rootdir

logging.basicConfig(stream=sys.stdout, format="%(message)s")

BACKUP_COUNT = 5
MAX_FILE_SIZE = 20000 * 1000  # bytes
LOGDIR = "logs"


def get_logger(name="blackopt", logdir=LOGDIR, **initial_values):
    path = os.path.join(get_rootdir(), logdir, name)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    def rotating_file_handler(log_path):
        return RotatingFileHandler(
            log_path,
            mode="a",
            maxBytes=BACKUP_COUNT * MAX_FILE_SIZE,
            backupCount=BACKUP_COUNT,
        )

    file_hdlr = rotating_file_handler(path)
    if os.path.exists(path):
        try:
            if os.stat(path).st_size != 0:
                file_hdlr.doRollover()
        except FileNotFoundError:
            # Rollover failed. prefer to overwrite old log instead terminating Moonfish.
            file_hdlr = rotating_file_handler(path)

    logger = logging.getLogger(name)
    logger.addHandler(file_hdlr)
    logger.setLevel(logging.INFO)

    log = wrap_logger(
        logger,
        processors=[
            # filter_by_level,
            TimeStamper(fmt="iso"),
            JSONRenderer(sort_keys=True),
        ],
        **initial_values
    )

    return log


if __name__ == "__main__":
    log = get_logger()
    log.info("something.filtered")
    log.warning("something.not_filtered")

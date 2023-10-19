import logging
import logging.config
import os

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {"default": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"}},
    "handlers": {
        "default": {
            "level": "DEBUG",
            "formatter": "default",
            "class": "logging.StreamHandler",
        }
    },
    "loggers": {
        "elexmodel": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": True,
        }
    },
}


def initialize_logging(logging_config=None):
    """
    Configures logging for the app.
    """
    if not logging_config:
        app_log_level = os.getenv("APP_LOG_LEVEL", "INFO")
        LOGGING_CONFIG["loggers"]["elexmodel"]["level"] = app_log_level
        logging_config = LOGGING_CONFIG
    logging.config.dictConfig(logging_config)
    logging.captureWarnings(True)

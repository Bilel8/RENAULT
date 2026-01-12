import sys
from loguru import logger
import logging

def configure_logging():
    """
    Configures loguru to replace standard logging and handle all logs.
    """
    # Remove default logger to avoid duplication
    logger.remove()

    # Add a sink to stderr with a nice format
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG",
    )

    # Intercept standard logging messages
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    # Configure standard logging to use InterceptHandler
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Silence uvicorn access logs to avoid duplicates if uvicorn uses its own logger
    # But often we want to see them.
    # By settings handlers=[], we basically remove existing handlers and force ours.
    
    # Specific adjustment for uvicorn to ensure it goes through loguru nicely
    for log_name in ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]:
        logging_logger = logging.getLogger(log_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False


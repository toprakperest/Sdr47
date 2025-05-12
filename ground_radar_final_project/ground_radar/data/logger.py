import logging
import os

class DataLogger:
    def __init__(self, log_file_name='system_log.log', log_level=logging.INFO):
        """
        Initializes the DataLogger class.

        Args:
            log_file_name (str): The name of the log file.
            log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
        """
        self.log_file_name = log_file_name
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Create a file handler
        handler = logging.FileHandler(self.log_file_name)
        handler.setLevel(log_level)

        # Create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(handler)

    def log(self, level, message):
        """
        Logs a message at the specified level.

        Args:
            level (int): The logging level (e.g., logging.INFO, logging.DEBUG, logging.ERROR).
            message (str): The message to log.
        """
        if level == logging.DEBUG:
            self.logger.debug(message)
        elif level == logging.INFO:
            self.logger.info(message)
        elif level == logging.WARNING:
            self.logger.warning(message)
        elif level == logging.ERROR:
            self.logger.error(message)
        elif level == logging.CRITICAL:
            self.logger.critical(message)
        else:
            self.logger.info(message)  # Default to info if level is unknown


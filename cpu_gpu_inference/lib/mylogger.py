from typing import List
import os
import torch
import datetime

class MyLogger:
    """
    My logger class for handling logging functionality.

    Attributes:
    - - - - - - - - - - - - - - - - - - - -
    name : str
        Name of the logger.
    level : int
        Level of the logger. Default is logging.DEBUG.
    handlers : List
        List of handlers associated with the logger. Default is None.
    """

    def __init__(self, logger_dir, name: str = 'logger', filename: str = 'mylog.log'):
        """
        Constructor for MyLogger class.

        Args:
        - - - - - - - - - - - - - - - - - - -
        name : str
            Name of the logger.
        filename : str, optional
            Name of the log file. Default is 'mylog.log'.
        """
        self.name = name
        self.filename = filename
        self.filename = os.path.join(logger_dir, self.filename)
        self.handlers = []

        # create a file handler
        file_handler = open(self.filename, mode='w')

        # add the file handler to the logger
        self.handlers.append(file_handler)
        
    def debug(self, message: str):
        """
        Logs a debug message.

        Args:
        - - - - - - - - - - - - - - - - - - -
        message : str
            Message to be logged.
        """
        self._log("DEBUG", message)

    def info(self, message: str):
        """
        Logs an info message.

        Args:
        - - - - - - - - - - - - - - - - - - -
        message : str
            Message to be logged.
        """
        self._log("INFO", message)

    def metric(self, message: str):
        """
        Logs an metric info message.

        Args:
        - - - - - - - - - - - - - - - - - - -
        message : str
            Message to be logged.
        """
        self._log("METRIC", message)

    def warning(self, message: str):
        """
        Logs a warning message.

        Args:
        - - - - - - - - - - - - - - - - - - -
        message : str
            Message to be logged.
        """
        self._log("WARNING", message)

    def error(self, message: str):
        """
        Logs an error message.

        Args:
        - - - - - - - - - - - - - - - - - - -
        message : str
            Message to be logged.
        """
        self._log("ERROR", message)

    def critical(self, message: str):
        """
        Logs a critical message.

        Args:
        - - - - - - - - - - - - - - - - - - -
        message : str
            Message to be logged.
        """
        self._log("CRITICAL", message)
    
    def _log(self, level: str, message: str):
        """
        Logs a message with the given level.

        Args:
        - - - - - - - - - - - - - - - - - - -
        level : str
            Logging level of the message.
        message : str
            Message to be logged.
        """
        now = datetime.datetime.now()
        log_time = now.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{log_time}] [{self.name}] [{level}] {message}\n"

        for handler in self.handlers:
            handler.write(log_message)
            handler.flush()

if __name__ == '__main__':
    pass
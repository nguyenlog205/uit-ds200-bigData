import logging
from pathlib import Path

class Logger:
    def __init__(
        self,
        log_name: str,
        logging_dir: str = 'logging',
    ):
        # --- Setup the directory and file path ---
        self.log_dir = Path(logging_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{log_name}.log"

        # --- Create the logger instance ---
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.DEBUG)

        # --- Avoid adding multiple handlers if the logger is re-initialized ---
        if not self.logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            # Saves logs to your directory
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # Prints logs to the terminal
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger
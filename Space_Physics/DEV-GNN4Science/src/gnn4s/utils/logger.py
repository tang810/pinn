import os
import logging
from types import SimpleNamespace
from datetime import datetime

class Logger:
    def __init__(self, log_dirt, log_name=None, log_level=logging.INFO):
        self.log_dirt = log_dirt
        self.log_name = log_name
        self.log_level = log_level

        if self.log_name==None:
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_name = f"{current_datetime}.log"
        
        os.makedirs(self.log_dirt, exist_ok=True)
        logging.basicConfig(filename=f'{self.log_dirt}/{self.log_name}', 
            format='%(asctime)s [%(levelname)s]: %(message)s', level=self.log_level)
        
        self.logger = logging.getLogger()

    def log_info(self, message):
        print(message)
        self.logger.info(message)

    def log_warning(self, message):
        print(message)
        self.logger.warning(message)

    def log_error(self, message):
        print(message)
        self.logger.error(message)

    def log_debug(self, message):
        print(message)
        self.logger.debug(message)
    
    def log_dict(self, dictionary, prefix=""):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.log_dict(value, prefix=f"{prefix}{key}.")
            elif isinstance(value, SimpleNamespace):
                self.log_dict(vars(value), prefix=f"{prefix}{key}.")
            else:
                self.log_info(f"{prefix}{key}: {value}")
        
    def log_config(self, config, prefix=""):
        self.log_info("configurations:")
        self.log_dict(config, prefix)
        self.log_info("")
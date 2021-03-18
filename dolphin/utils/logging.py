import logging.config
import logging
from .config import load_config


class Logger():

    _init = {}

    def init_logger(self, cfg):
        
        logger_config = load_config(cfg['config_path'])['config']
        logger_name = cfg['logger_name']

        if logger_name not in self._init.keys():
            filename = cfg['filename']
            console_level = cfg['console_level']
            file_level = cfg['file_level']
            
            logger_config['handlers']['console']['level'] = console_level
            if filename is not None and file_level is not None:
                logger_config['handlers']['file']['level'] = file_level
                logger_config['handlers']['file']['filename'] = filename
                loggers = logger_config['loggers']
                logger_config['loggers'][logger_name] = loggers['FileLogger']
            else:
                loggers = logger_config['loggers']
                logger_config['loggers'][logger_name] = loggers['StreamLogger']

            logging.config.dictConfig(logger_config)

            self._init[logger_name] = logging.getLogger(logger_name)

    def get(self, logger_name=None):
        if logger_name is None:
            if len(self._init) == 1:
                for logger_name, logger in self._init.items():
                    return logger
            else:
                raise ValueError(f"There is not only one logger registered."
                                 " Please assign the specified logger name or"
                                 " registered it.")
        else:
            if logger_name not in self._init.keys():
                raise ValueError(f"Logger: {logger_name} not found.")
            return self._init[logger_name]

logger = Logger()
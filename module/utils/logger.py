import logging
import coloredlogs

def get_logger(module_name):
    logger = logging.getLogger(module_name)
    
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        coloredlogs.install(
            level='DEBUG',
            logger=logger,
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    return logger
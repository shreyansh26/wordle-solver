import os
import logging
from datetime import datetime
import pytz

def ist_time_converter(timestamp):
    return datetime.fromtimestamp(timestamp, pytz.timezone('Asia/Kolkata')).timetuple()

def get_logger(filename):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)

    # Check if logger already has handlers to avoid duplicates
    if logger.handlers:
        return logger

    if not os.path.exists('logs'):
        os.makedirs('logs')

    file_handler = logging.FileHandler(f'logs/{filename}.log')
    stream_handler = logging.StreamHandler()

    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_formatter.converter = ist_time_converter
    file_handler.setFormatter(log_formatter)
    stream_handler.setFormatter(log_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
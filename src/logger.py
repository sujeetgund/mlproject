import logging
import os
from datetime import datetime

# Create a unique file name
LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create directory where log files will be saved
DIR_PATH = os.path.join(os.getcwd(), 'logs', LOG_FILE_NAME)
os.makedirs(DIR_PATH, exist_ok=True)

# Create a log file path
LOG_FILE_PATH = os.path.join(DIR_PATH, LOG_FILE_NAME)

# Logging config
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

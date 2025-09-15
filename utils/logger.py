import logging
import os
from datetime import datetime

# Create experiments/logs directory if it doesn't exist
logs_dir = os.path.join('experiments', 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Set up logger with both console and file handler
logger = logging.getLogger('Autoformer_anomaly')
logger.setLevel(logging.INFO)

# Clear any existing handlers to avoid duplicate logs
if logger.hasHandlers():
    logger.handlers.clear()

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create file handler with timestamp in filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"anomaly_detection_{timestamp}.log"
file_handler = logging.FileHandler(os.path.join(logs_dir, log_filename))
file_handler.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)



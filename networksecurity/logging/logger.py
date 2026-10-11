import logging
import os
from datetime import datetime

# Generate a daily log filename structure
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define logs folder inside your root project directory
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(os.path.dirname(logs_path), exist_ok=True)

# Configure the universal logging engine
logging.basicConfig(
    filename=logs_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# CRUCIAL FIX: Export the configured logging instance as 'logger'
logger = logging.getLogger("networksecurity")

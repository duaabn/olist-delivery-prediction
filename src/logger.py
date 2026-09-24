import logging
import sys
from pathlib import Path

from src.config import config


log_path = Path(config["paths"]["log_file"])
log_path.parent.mkdir(parents=True, exist_ok=True)

# Logger
logger = logging.getLogger("olist_inference")
logger.setLevel(logging.INFO)

#(Log Format)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#(Console)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

#(File)
file_handler = logging.FileHandler(log_path, encoding="utf-8")
file_handler.setFormatter(formatter)

#Handlers
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
import logging
import os
import torch
from .config import BASE_DIR

# Setup log directory
log_dir = BASE_DIR
log_file = os.path.join(log_dir, "training.log")

# Setup logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d]: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode="a", encoding="utf-8")
    ]
)

logger = logging.getLogger("1bit_celeba")

def log_vram(step_name=""):
    """Logs the currently allocated VRAM on the default GPU if available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        logger.info(f"VRAM [{step_name}] -> Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")
    else:
        logger.debug(f"VRAM [{step_name}] -> CPU Mode (No VRAM)")

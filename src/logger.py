"""
Structured Logging — Credit Scoring System.

Uses loguru for structured, colored, file-rotated logging.
All modules import `logger` from here for consistent formatting.

Usage:
    from src.logger import logger
    logger.info("Model loaded successfully")
    logger.error("Failed to load model", exc_info=True)
"""

import sys
from loguru import logger as _logger

from src.config import LOGS_DIR

# Remove default handler
_logger.remove()

# ──────────────────────────────────────────────
# Console Handler — colored, human-readable
# ──────────────────────────────────────────────
_logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    ),
    level="DEBUG",
    colorize=True,
)

# ──────────────────────────────────────────────
# File Handler — JSON-structured, auto-rotated
# ──────────────────────────────────────────────
LOGS_DIR.mkdir(parents=True, exist_ok=True)

_logger.add(
    LOGS_DIR / "credit_scoring_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
    level="INFO",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
    encoding="utf-8",
)

# Export as `logger` for clean imports
logger = _logger

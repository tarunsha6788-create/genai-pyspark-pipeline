"""Configuration settings and environment helpers for the E-commerce Data Pipeline."""

import logging
import sys
from pathlib import Path
from typing import Sequence

# ---------------------------------------------------------------------------
# Directory Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

# File Paths for Raw Synthetic Datasets (CSV & Parquet)
CUSTOMERS_CSV: Path = RAW_DATA_DIR / "customers.csv"
PRODUCTS_CSV: Path = RAW_DATA_DIR / "products.csv"
ORDERS_CSV: Path = RAW_DATA_DIR / "orders.csv"

CUSTOMERS_PARQUET: Path = RAW_DATA_DIR / "customers.parquet"
PRODUCTS_PARQUET: Path = RAW_DATA_DIR / "products.parquet"
ORDERS_PARQUET: Path = RAW_DATA_DIR / "orders.parquet"

# ---------------------------------------------------------------------------
# Synthetic Data Generation Settings (Defaults)
# ---------------------------------------------------------------------------
DEFAULT_NUM_CUSTOMERS: int = 100_000
DEFAULT_NUM_PRODUCTS: int = 10_000
DEFAULT_NUM_ORDERS: int = 1_000_000

PRODUCT_CATEGORIES: Sequence[str] = (
    "Electronics",
    "Clothing",
    "Home",
    "Sports",
    "Books",
)

ORDER_STATUSES: Sequence[str] = (
    "Completed",
    "Cancelled",
    "Returned",
    "Processing",
)
ORDER_STATUS_WEIGHTS: Sequence[float] = (0.75, 0.10, 0.05, 0.10)

PAYMENT_METHODS: Sequence[str] = (
    "Credit Card",
    "Debit Card",
    "PayPal",
    "UPI",
    "Bank Transfer",
)

# ---------------------------------------------------------------------------
# Spark Session Configuration
# ---------------------------------------------------------------------------
SPARK_APP_NAME: str = "EcommerceDataPipeline"
SPARK_MASTER: str = "local[*]"
SPARK_DRIVER_MEMORY: str = "2g"


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
def setup_logging(
    name: str = "ecommerce_pipeline",
    level: int = logging.INFO
) -> logging.Logger:
    """Configures and returns a standardized logger for pipeline modules.

    Args:
        name: Name of the logger instance (defaults to 'ecommerce_pipeline').
        level: Logging severity level (defaults to logging.INFO).

    Returns:
        A configured logging.Logger instance with formatted console output.
    """
    logger: logging.Logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)
        handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter: logging.Formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


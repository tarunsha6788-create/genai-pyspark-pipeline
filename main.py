"""Main orchestrator script for synthetic e-commerce dataset generation.

Generates customers, products, and orders using SyntheticDataGenerator,
persists them as Parquet files into data/raw/, and reports runtime and file metrics.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src import config
from src.data_generator import SyntheticDataGenerator

logger: logging.Logger = config.setup_logging(name="main_pipeline")


def format_bytes(size_bytes: int) -> str:
    """Converts a byte count into a human-readable string (KB, MB, GB).

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted string (e.g., '14.25 MB').
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    for unit in ["KB", "MB", "GB", "TB"]:
        size_bytes_float = size_bytes / 1024.0
        if size_bytes_float < 1024.0 or unit == "TB":
            return f"{size_bytes_float:.2f} {unit}"
        size_bytes = int(size_bytes_float)
    return f"{size_bytes} B"


def save_as_parquet(df: pd.DataFrame, file_path: Path) -> Tuple[float, int]:
    """Saves a pandas DataFrame to a Parquet file and returns write time and file size.

    Args:
        df: Pandas DataFrame to serialize.
        file_path: Destination path for the Parquet file.

    Returns:
        A tuple of (write_duration_seconds, file_size_in_bytes).

    Raises:
        ImportError: If neither pyarrow nor fastparquet is installed.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Serializing %d rows to Parquet at '%s'...", len(df), file_path)

    start_write: float = time.perf_counter()
    try:
        df.to_parquet(file_path, engine="pyarrow", index=False, compression="snappy")
    except ImportError as err:
        logger.error(
            "PyArrow or fastparquet engine is missing. Install pyarrow via 'pip install pyarrow'."
        )
        raise err

    duration: float = time.perf_counter() - start_write
    file_size: int = file_path.stat().st_size
    return duration, file_size


def run_synthetic_pipeline() -> None:
    """Executes end-to-end synthetic data generation and Parquet persistence.

    Includes timing benchmarks, file size calculation, and error handling.
    """
    pipeline_start: float = time.perf_counter()
    metrics: Dict[str, Dict[str, str]] = {}

    print("=" * 70)
    print("      E-COMMERCE DATA PIPELINE: SYNTHETIC DATA GENERATION")
    print("=" * 70)
    logger.info("Initializing synthetic pipeline execution...")

    try:
        # Instantiate generator using settings from config
        generator = SyntheticDataGenerator(seed=42, output_dir=config.RAW_DATA_DIR)

        # -------------------------------------------------------------------
        # 1. Customers Generation
        # -------------------------------------------------------------------
        print(f"\n[1/3] Generating {config.DEFAULT_NUM_CUSTOMERS:,} Customers...")
        cust_gen_start: float = time.perf_counter()
        customers_df: pd.DataFrame = generator.generate_customers(
            num_customers=config.DEFAULT_NUM_CUSTOMERS
        )
        cust_gen_time: float = time.perf_counter() - cust_gen_start

        cust_write_time, cust_file_size = save_as_parquet(
            customers_df, config.CUSTOMERS_PARQUET
        )
        metrics["Customers"] = {
            "Rows": f"{len(customers_df):,}",
            "Gen Time": f"{cust_gen_time:.2f}s",
            "Write Time": f"{cust_write_time:.2f}s",
            "File Size": format_bytes(cust_file_size),
            "File Path": str(config.CUSTOMERS_PARQUET),
        }

        # -------------------------------------------------------------------
        # 2. Products Generation
        # -------------------------------------------------------------------
        print(f"\n[2/3] Generating {config.DEFAULT_NUM_PRODUCTS:,} Products...")
        prod_gen_start: float = time.perf_counter()
        products_df: pd.DataFrame = generator.generate_products(
            num_products=config.DEFAULT_NUM_PRODUCTS
        )
        prod_gen_time: float = time.perf_counter() - prod_gen_start

        prod_write_time, prod_file_size = save_as_parquet(
            products_df, config.PRODUCTS_PARQUET
        )
        metrics["Products"] = {
            "Rows": f"{len(products_df):,}",
            "Gen Time": f"{prod_gen_time:.2f}s",
            "Write Time": f"{prod_write_time:.2f}s",
            "File Size": format_bytes(prod_file_size),
            "File Path": str(config.PRODUCTS_PARQUET),
        }

        # -------------------------------------------------------------------
        # 3. Orders Generation (Pareto 80/20 Rule)
        # -------------------------------------------------------------------
        print(f"\n[3/3] Generating {config.DEFAULT_NUM_ORDERS:,} Orders (Pareto Distribution)...")
        ord_gen_start: float = time.perf_counter()
        orders_df: pd.DataFrame = generator.generate_orders(
            customers_df=customers_df,
            products_df=products_df,
            num_orders=config.DEFAULT_NUM_ORDERS,
            pareto_alpha=1.16,
        )
        ord_gen_time: float = time.perf_counter() - ord_gen_start

        ord_write_time, ord_file_size = save_as_parquet(
            orders_df, config.ORDERS_PARQUET
        )
        metrics["Orders"] = {
            "Rows": f"{len(orders_df):,}",
            "Gen Time": f"{ord_gen_time:.2f}s",
            "Write Time": f"{ord_write_time:.2f}s",
            "File Size": format_bytes(ord_file_size),
            "File Path": str(config.ORDERS_PARQUET),
        }

        # -------------------------------------------------------------------
        # Pipeline Summary Report
        # -------------------------------------------------------------------
        total_time: float = time.perf_counter() - pipeline_start
        total_size_bytes: int = cust_file_size + prod_file_size + ord_file_size

        print("\n" + "=" * 70)
        print("                  GENERATION SUMMARY REPORT")
        print("=" * 70)
        header = f"{'Dataset':<12} | {'Rows':<11} | {'Gen Time':<10} | {'Write Time':<10} | {'File Size':<10}"
        print(header)
        print("-" * len(header))

        for name, item in metrics.items():
            print(
                f"{name:<12} | {item['Rows']:<11} | {item['Gen Time']:<10} | {item['Write Time']:<10} | {item['File Size']:<10}"
            )

        print("-" * len(header))
        print(f"Total Disk Footprint: {format_bytes(total_size_bytes)}")
        print(f"Total Elapsed Time:   {total_time:.2f}s ({total_time / 60.0:.2f} min)")
        print("=" * 70)

        logger.info(
            "Pipeline successfully completed in %.2fs. Total Parquet storage: %s",
            total_time,
            format_bytes(total_size_bytes),
        )

    except KeyboardInterrupt:
        logger.warning("\nPipeline process interrupted by user.")
        sys.exit(130)

    except ImportError as imp_err:
        logger.critical("Required dependency is not installed: %s", imp_err)
        print(f"\n[ERROR] Missing dependency: {imp_err}")
        print("Please install required packages using: pip install pyarrow")
        sys.exit(1)

    except Exception as exc:
        logger.exception("An unhandled exception occurred during data generation: %s", exc)
        print(f"\n[ERROR] Pipeline failed with error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    run_synthetic_pipeline()


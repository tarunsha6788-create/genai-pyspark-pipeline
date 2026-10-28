"""Standalone execution script for running PySpark SalesAnalytics.

Loads raw Parquet datasets from data/raw/, executes core analytical queries
(top customers, category aggregations, and monthly trends with Window functions),
displays results in formatted console tables, benchmarks execution times,
and safely tears down the SparkSession.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame, SparkSession

from src import config
from src.spark_analytics import SalesAnalytics

logger: logging.Logger = config.setup_logging(name="run_analytics")


def check_prerequisites() -> bool:
    """Verifies that required Parquet datasets exist in data/raw/.

    Returns:
        True if all files exist, False otherwise.
    """
    required_files = [
        ("Orders", config.ORDERS_PARQUET),
        ("Products", config.PRODUCTS_PARQUET),
        ("Customers", config.CUSTOMERS_PARQUET),
    ]

    missing = [name for name, path in required_files if not path.exists()]
    if missing:
        print("\n" + "!" * 75)
        print(f"[ERROR] Missing Parquet datasets in '{config.RAW_DATA_DIR}': {', '.join(missing)}")
        print("Please generate the datasets first by running:")
        print("    python main.py")
        print("!" * 75 + "\n")
        return False
    return True


def main() -> None:
    """Orchestrates the PySpark SalesAnalytics workflow with performance timing."""
    if not check_prerequisites():
        sys.exit(1)

    print("=" * 80)
    print("               PYSPARK E-COMMERCE SALES ANALYTICS")
    print("=" * 80)

    total_start: float = time.perf_counter()
    analytics: Optional[SalesAnalytics] = None

    try:
        # -------------------------------------------------------------------
        # 1 & 2. Initialize SparkSession via SalesAnalytics
        # -------------------------------------------------------------------
        print("\n[Step 1/4] Initializing SparkSession (4GB, AQE=True, KryoSerializer)...")
        spark_init_start: float = time.perf_counter()
        spark: SparkSession = SalesAnalytics.create_spark_session(
            app_name="EcommerceSalesAnalytics",
            memory="4g",
        )
        analytics = SalesAnalytics(spark=spark)
        spark_init_time: float = time.perf_counter() - spark_init_start
        print(f"--> SparkSession ready in {spark_init_time:.2f}s")

        # -------------------------------------------------------------------
        # 3. Load Parquet Files from data/raw/
        # -------------------------------------------------------------------
        print(f"\n[Step 2/4] Loading Parquet datasets from '{config.RAW_DATA_DIR}'...")
        load_start: float = time.perf_counter()

        customers_df: DataFrame = analytics.load_parquet(config.CUSTOMERS_PARQUET)
        products_df: DataFrame = analytics.load_parquet(config.PRODUCTS_PARQUET)
        orders_df: DataFrame = analytics.load_parquet(config.ORDERS_PARQUET)

        # Trigger quick count to materialize metadata and report record counts
        cust_count: int = customers_df.count()
        prod_count: int = products_df.count()
        ord_count: int = orders_df.count()
        load_time: float = time.perf_counter() - load_start

        print(f"--> Loaded Customers: {cust_count:,} records")
        print(f"--> Loaded Products:  {prod_count:,} records")
        print(f"--> Loaded Orders:    {ord_count:,} records")
        print(f"--> Parquet Ingestion completed in {load_time:.2f}s")

        # -------------------------------------------------------------------
        # 4 & 5 & 6. Run Analytics Methods and Display Results
        # -------------------------------------------------------------------
        print("\n[Step 3/4] Executing Analytical Pipelines & Displaying Results...\n")

        # Query A: Top 10 Customers by Revenue
        print("-" * 80)
        print("QUERY 1: Top 10 VIP Customers by Gross Revenue")
        print("-" * 80)
        q1_start: float = time.perf_counter()
        top_customers_df: DataFrame = analytics.top_customers_by_revenue(
            orders_df=orders_df,
            products_df=products_df,
            n=10,
            customers_df=customers_df,
        )
        top_customers_df.show(truncate=False)
        q1_time: float = time.perf_counter() - q1_start
        print(f"--> Query 1 Execution & Display Time: {q1_time:.2f}s\n")

        # Query B: Sales by Category
        print("-" * 80)
        print("QUERY 2: Sales & Revenue Breakdown by Product Category")
        print("-" * 80)
        q2_start: float = time.perf_counter()
        category_sales_df: DataFrame = analytics.sales_by_category(
            orders_df=orders_df,
            products_df=products_df,
        )
        category_sales_df.show(truncate=False)
        q2_time: float = time.perf_counter() - q2_start
        print(f"--> Query 2 Execution & Display Time: {q2_time:.2f}s\n")

        # Query C: Monthly Trends with MoM Growth (Window Function)
        print("-" * 80)
        print("QUERY 3: Monthly Trends & Month-over-Month Revenue Growth % (Window Function)")
        print("-" * 80)
        q3_start: float = time.perf_counter()
        monthly_trends_df: DataFrame = analytics.monthly_trends(
            orders_df=orders_df,
            products_df=products_df,
        )
        monthly_trends_df.show(36, truncate=False)
        q3_time: float = time.perf_counter() - q3_start
        print(f"--> Query 3 Execution & Display Time: {q3_time:.2f}s\n")

        # -------------------------------------------------------------------
        # Execution Summary
        # -------------------------------------------------------------------
        total_time: float = time.perf_counter() - total_start
        print("=" * 80)
        print("                      ANALYTICS TIMING SUMMARY")
        print("=" * 80)
        print(f"{'Operation':<45} | {'Duration':>10}")
        print("-" * 80)
        print(f"{'SparkSession Startup':<45} | {spark_init_time:>9.2f}s")
        print(f"{'Parquet Data Loading':<45} | {load_time:>9.2f}s")
        print(f"{'Top Customers by Revenue':<45} | {q1_time:>9.2f}s")
        print(f"{'Sales by Product Category':<45} | {q2_time:>9.2f}s")
        print(f"{'Monthly Trends & MoM Growth (Window)':<45} | {q3_time:>9.2f}s")
        print("-" * 80)
        print(f"{'Total Elapsed Time':<45} | {total_time:>9.2f}s ({total_time / 60.0:.2f} min)")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n[WARNING] Analytics execution interrupted by user.")
        logger.warning("Pipeline interrupted by user.")
        sys.exit(130)

    except Exception as exc:
        logger.exception("An error occurred during analytics execution: %s", exc)
        print(f"\n[ERROR] Analytics pipeline failed: {exc}")
        sys.exit(1)

    finally:
        # -------------------------------------------------------------------
        # 7. Stop SparkSession
        # -------------------------------------------------------------------
        if analytics is not None:
            print("\n[Step 4/4] Stopping SparkSession...")
            analytics.stop()
            print("--> SparkSession terminated cleanly.")


if __name__ == "__main__":
    main()


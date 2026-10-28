"""File format benchmarking suite with hardware and energy metrics.

Benchmarks CSV, XLSX, Parquet (PyArrow & fastparquet), ORC, and Feather formats
across file size, read/write latency, peak RAM consumption (tracemalloc),
CPU execution time, and estimated electrical energy consumption.
"""

from dataclasses import dataclass
from datetime import datetime
import gc
import logging
import os
from pathlib import Path
import time
import tracemalloc
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow  # noqa: F401
import fastparquet  # noqa: F401

from src import config

logger: logging.Logger = config.setup_logging(name="format_benchmark")

# Constants
CPU_TDP_WATTS: float = 65.0  # Standard desktop/laptop TDP rating in Watts
BENCHMARK_TEMP_DIR: Path = config.DATA_DIR / "benchmark_temp"


@dataclass
class BenchmarkMetric:
    """Holds individual performance metrics for a specific file format."""

    format_name: str
    file_size_mb: float
    write_time_s: float
    read_time_s: float
    peak_memory_mb: float
    total_cpu_time_s: float
    energy_consumption_wh: float
    error: Optional[str] = None


def generate_benchmark_dataframe(num_rows: int = 500_000) -> pd.DataFrame:
    """Rapidly generates a synthetic DataFrame with 500,000 records.

    Columns: id, name, email, amount, date, category

    Args:
        num_rows: Number of rows to generate (default: 500,000).

    Returns:
        A populated pandas DataFrame.
    """
    logger.info("Synthesizing test DataFrame with %d rows...", num_rows)
    np.random.seed(42)

    # 1. IDs
    ids = np.arange(1, num_rows + 1, dtype=np.int64)

    # 2. Names & Emails using vector pooling for high-speed synthesis
    first_names = [
        "Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Pat", "Robin",
        "Casey", "Jamie", "Riley", "Cameron", "Jesse", "Dakota", "Logan",
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson",
    ]
    first_choice = np.random.choice(first_names, size=num_rows)
    last_choice = np.random.choice(last_names, size=num_rows)
    names = [f"{f} {l}" for f, l in zip(first_choice, last_choice)]
    emails = [
        f"{f.lower()}.{l.lower()}.{idx}@example.com"
        for idx, (f, l) in enumerate(zip(first_choice, last_choice), start=1)
    ]

    # 3. Monetary Amount
    amounts = np.round(np.random.uniform(5.0, 1500.0, size=num_rows), 2)

    # 4. Dates (between 2022-01-01 and 2025-01-01)
    start_ts = int(datetime(2022, 1, 1).timestamp())
    end_ts = int(datetime(2025, 1, 1).timestamp())
    random_epochs = np.random.randint(start_ts, end_ts, size=num_rows, dtype=np.int64)
    dates = [datetime.fromtimestamp(ts).strftime("%Y-%m-%d") for ts in random_epochs]

    # 5. Categories
    categories_pool = ["Electronics", "Clothing", "Home", "Sports", "Books"]
    categories = np.random.choice(categories_pool, size=num_rows)

    df = pd.DataFrame({
        "id": ids,
        "name": names,
        "email": emails,
        "amount": amounts,
        "date": dates,
        "category": categories,
    })

    logger.info("DataFrame created in memory. Size: %.2f MB", df.memory_usage(deep=True).sum() / (1024 * 1024))
    return df


def measure_operation(operation_fn: Callable[[], None]) -> Tuple[float, float, float]:
    """Measures wall-clock time, CPU process time, and peak memory of a callable.

    Args:
        operation_fn: Function to execute and measure.

    Returns:
        Tuple of (elapsed_wall_time_s, cpu_process_time_s, peak_memory_mb).
    """
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()

    start_wall: float = time.perf_counter()
    start_cpu: float = time.process_time()

    operation_fn()

    cpu_time: float = time.process_time() - start_cpu
    wall_time: float = time.perf_counter() - start_wall

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()

    peak_memory_mb: float = peak_bytes / (1024.0 * 1024.0)
    return wall_time, cpu_time, peak_memory_mb


def benchmark_single_format(
    format_name: str,
    file_path: Path,
    write_fn: Callable[[pd.DataFrame, Path], None],
    read_fn: Callable[[Path], pd.DataFrame],
    df: pd.DataFrame,
) -> BenchmarkMetric:
    """Benchmarks write and read performance for a single file format.

    Args:
        format_name: Display name of the file format.
        file_path: Path to the target file.
        write_fn: Function to save the DataFrame to file_path.
        read_fn: Function to read the DataFrame from file_path.
        df: Input DataFrame to serialize.

    Returns:
        A BenchmarkMetric instance populated with collected measurements.
    """
    logger.info("--> Benchmarking: %s...", format_name)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Benchmark Write
        write_wall, write_cpu, write_peak_mem = measure_operation(
            lambda: write_fn(df, file_path)
        )

        # 2. Measure File Size
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = file_size_bytes / (1024.0 * 1024.0)

        # 3. Benchmark Read
        read_wall, read_cpu, read_peak_mem = measure_operation(
            lambda: read_fn(file_path)
        )

        # 4. Total CPU Time and Estimated Energy (Watt-Hours)
        # Energy (Wh) = (Total CPU time in seconds * TDP Watts) / 3600 seconds
        total_cpu = write_cpu + read_cpu
        energy_wh = (total_cpu * CPU_TDP_WATTS) / 3600.0
        overall_peak_mem = max(write_peak_mem, read_peak_mem)

        logger.info(
            "[%s] Size: %.2f MB | Write: %.2fs | Read: %.2fs | Peak RAM: %.2f MB | Energy: %.5f Wh",
            format_name, file_size_mb, write_wall, read_wall, overall_peak_mem, energy_wh
        )

        return BenchmarkMetric(
            format_name=format_name,
            file_size_mb=file_size_mb,
            write_time_s=write_wall,
            read_time_s=read_wall,
            peak_memory_mb=overall_peak_mem,
            total_cpu_time_s=total_cpu,
            energy_consumption_wh=energy_wh,
        )

    except Exception as exc:
        logger.exception("Failed to benchmark format %s: %s", format_name, exc)
        return BenchmarkMetric(
            format_name=format_name,
            file_size_mb=0.0,
            write_time_s=0.0,
            read_time_s=0.0,
            peak_memory_mb=0.0,
            total_cpu_time_s=0.0,
            energy_consumption_wh=0.0,
            error=str(exc),
        )


def print_comparison_tables(metrics: List[BenchmarkMetric], baseline_format: str = "CSV") -> None:
    """Formats and prints standard comparison and relative savings tables.

    Args:
        metrics: List of benchmark results.
        baseline_format: Format name to compare against for percentage savings.
    """
    valid_metrics = [m for m in metrics if m.error is None]
    baseline: Optional[BenchmarkMetric] = next(
        (m for m in valid_metrics if m.format_name.upper() == baseline_format.upper()), None
    )

    # -----------------------------------------------------------------------
    # Table 1: Raw Performance & Hardware Metrics
    # -----------------------------------------------------------------------
    print("\n" + "=" * 105)
    print("                      FILE FORMAT BENCHMARK RESULTS (500,000 ROWS)")
    print(f"       Hardware Assumption: {CPU_TDP_WATTS:.0f}W CPU TDP | Peak RAM: tracemalloc | CPU Time: process_time")
    print("=" * 105)

    headers_raw = [
        "Format",
        "Size (MB)",
        "Write (s)",
        "Read (s)",
        "Peak RAM (MB)",
        "CPU Time (s)",
        "Energy (Wh)",
    ]
    row_fmt = "{:<24} | {:>10} | {:>10} | {:>10} | {:>13} | {:>12} | {:>12}"
    print(row_fmt.format(*headers_raw))
    print("-" * 105)

    for m in metrics:
        if m.error:
            print(f"{m.format_name:<24} | ERROR: {m.error}")
        else:
            print(
                row_fmt.format(
                    m.format_name,
                    f"{m.file_size_mb:.2f}",
                    f"{m.write_time_s:.2f}",
                    f"{m.read_time_s:.2f}",
                    f"{m.peak_memory_mb:.2f}",
                    f"{m.total_cpu_time_s:.2f}",
                    f"{m.energy_consumption_wh:.5f}",
                )
            )
    print("=" * 105)

    # -----------------------------------------------------------------------
    # Table 2: Percentage Savings vs CSV Baseline
    # -----------------------------------------------------------------------
    if not baseline:
        logger.warning("Baseline '%s' was not evaluated. Skipping savings table.", baseline_format)
        return

    print("\n" + "=" * 105)
    print(f"                   PERCENTAGE SAVINGS VS {baseline.format_name} BASELINE")
    print("       Formula: (Baseline - Format) / Baseline * 100%  [Positive = Savings / Speedup]")
    print("=" * 105)

    headers_sav = [
        "Format",
        "Size Saved",
        "Write Speedup",
        "Read Speedup",
        "RAM Saved",
        "CPU Saved",
        "Energy Saved",
    ]
    row_sav_fmt = "{:<24} | {:>11} | {:>13} | {:>12} | {:>11} | {:>11} | {:>12}"
    print(row_sav_fmt.format(*headers_sav))
    print("-" * 105)

    def calc_pct(base_val: float, current_val: float) -> str:
        if base_val <= 0.0:
            return "N/A"
        pct = ((base_val - current_val) / base_val) * 100.0
        prefix = "+" if pct > 0 else ""
        return f"{prefix}{pct:.1f}%"

    for m in valid_metrics:
        size_pct = calc_pct(baseline.file_size_mb, m.file_size_mb)
        write_pct = calc_pct(baseline.write_time_s, m.write_time_s)
        read_pct = calc_pct(baseline.read_time_s, m.read_time_s)
        ram_pct = calc_pct(baseline.peak_memory_mb, m.peak_memory_mb)
        cpu_pct = calc_pct(baseline.total_cpu_time_s, m.total_cpu_time_s)
        energy_pct = calc_pct(baseline.energy_consumption_wh, m.energy_consumption_wh)

        print(
            row_sav_fmt.format(
                m.format_name,
                size_pct,
                write_pct,
                read_pct,
                ram_pct,
                cpu_pct,
                energy_pct,
            )
        )
    print("=" * 105)


def run_all_benchmarks(num_rows: int = 500_000, cleanup: bool = True) -> List[BenchmarkMetric]:
    """Runs end-to-end benchmarks across CSV, XLSX, Parquet, ORC, and Feather formats.

    Args:
        num_rows: Row count for the benchmark dataset (default: 500,000).
        cleanup: Whether to delete temporary benchmark files after evaluation.

    Returns:
        List of BenchmarkMetric objects.
    """
    # 1. Synthesize input DataFrame
    df = generate_benchmark_dataframe(num_rows=num_rows)
    temp_dir = BENCHMARK_TEMP_DIR
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 2. Format Definitions & Serializers
    # Note: XLSX write for 500k rows can take several minutes due to XML compression
    format_suites: Dict[str, Tuple[Path, Callable[[pd.DataFrame, Path], None], Callable[[Path], pd.DataFrame]]] = {
        "CSV": (
            temp_dir / "bench.csv",
            lambda d, p: d.to_csv(p, index=False),
            lambda p: pd.read_csv(p),
        ),
        "Parquet (PyArrow)": (
            temp_dir / "bench_arrow.parquet",
            lambda d, p: d.to_parquet(p, engine="pyarrow", compression="snappy", index=False),
            lambda p: pd.read_parquet(p, engine="pyarrow"),
        ),
        "Parquet (fastparquet)": (
            temp_dir / "bench_fast.parquet",
            lambda d, p: d.to_parquet(p, engine="fastparquet", compression="snappy", index=False),
            lambda p: pd.read_parquet(p, engine="fastparquet"),
        ),
        "ORC": (
            temp_dir / "bench.orc",
            lambda d, p: d.to_orc(p, engine="pyarrow", index=False),
            lambda p: pd.read_orc(p),
        ),
        "Feather": (
            temp_dir / "bench.feather",
            lambda d, p: d.to_feather(p),
            lambda p: pd.read_feather(p),
        ),
        "XLSX (openpyxl)": (
            temp_dir / "bench.xlsx",
            lambda d, p: d.to_excel(p, engine="openpyxl", index=False),
            lambda p: pd.read_excel(p, engine="openpyxl"),
        ),
    }

    results: List[BenchmarkMetric] = []

    print("\nStarting benchmarks across file formats...")
    for format_name, (file_path, write_fn, read_fn) in format_suites.items():
        metric = benchmark_single_format(
            format_name=format_name,
            file_path=file_path,
            write_fn=write_fn,
            read_fn=read_fn,
            df=df,
        )
        results.append(metric)

    # 3. Print Results
    print_comparison_tables(results, baseline_format="CSV")

    # 4. Optional cleanup
    if cleanup:
        logger.info("Cleaning up temporary benchmark files in '%s'...", temp_dir)
        for _, (file_path, _, _) in format_suites.items():
            if file_path.exists():
                try:
                    file_path.unlink()
                except OSError as err:
                    logger.warning("Could not delete %s: %s", file_path, err)

    return results


if __name__ == "__main__":
    run_all_benchmarks(num_rows=500_000, cleanup=True)


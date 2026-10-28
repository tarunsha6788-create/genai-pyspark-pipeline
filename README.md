# E-Commerce Data Pipeline with PySpark

An end-to-end e-commerce data engineering pipeline in Python. This project generates realistic synthetic transactional data (customers, products, orders) and performs large-scale analytics using **Apache PySpark** to extract actionable business insights.

---

## Project Structure

```text
genai-pyspark-pipeline/
│
├── data/
│   ├── raw/                  # Generated synthetic CSV datasets (customers, products, orders)
│   └── processed/            # PySpark analytical output reports
│
├── notebooks/                # Jupyter Notebooks for exploratory data analysis
│
├── src/                      # Source code for the data pipeline
│   ├── __init__.py           # Package initializer
│   ├── config.py             # Global configurations, file paths, and logger setup
│   ├── data_generator.py     # Faker-based synthetic data generator
│   └── spark_analytics.py    # PySpark ETL and analytical queries
│
├── tests/                    # Unit and integration test suite
│   ├── __init__.py
│   └── test_data_generator.py
│
├── .gitignore                # Git ignore rules for Python, data files, and Spark
├── benchmark_formats.py      # Benchmark CSV, XLSX, Parquet, ORC, Feather with hardware metrics
├── compare_pandas_pyspark.py # Direct performance benchmark: Pandas vs PySpark on 1M rows
├── main.py                   # Orchestrator to generate and persist Parquet datasets
├── run_analytics.py          # PySpark runner with timing for Top Customers, Category & MoM Growth
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation and guide
```

---

## Features

- **Synthetic Data Generation (`src/data_generator.py`)**:
  - Uses `Faker` to produce deterministic, realistic e-commerce datasets.
  - Enforces relational referential integrity (orders link valid `customer_id` and `product_id`).
  - Realistic product pricing by retail category, chronological ordering logic, and order status weighting.

- **PySpark Analytics Engine (`src/spark_analytics.py`)**:
  - **Category Performance**: Evaluates revenue, order volume, and Average Order Value (AOV) per product category.
  - **Top High-Value Customers**: Identifies VIP customers ranked by completed transactional spend.
  - **Monthly Sales Velocity**: Analyzes monthly revenue progression and unique active customer trends.
  - **Order Status Breakdown**: Computes distribution and cancellation rates across order lifecycles.
  - **Product Performance**: Highlights best-selling inventory by total revenue and unit sales.

- **Production-Ready Python Code**:
  - Full Python type hints (`typing`).
  - PEP 257 compliant docstrings.
  - Standardized application logging with timestamps and log levels.

---

## Prerequisites

1. **Python 3.9+**
2. **Java 8, 11, or 17** (Required by Apache Spark)
   - On Windows, install Eclipse Temurin JDK 17:
     ```powershell
     winget install EclipseAdoptium.Temurin.17.JDK
     ```
   - Ensure the `JAVA_HOME` environment variable is configured and pointing to your JDK directory:
     ```powershell
     $env:JAVA_HOME
     ```

---

## Installation & Setup

1. **Clone or Navigate to the Project Directory**:
   ```bash
   cd genai-pyspark-pipeline
   ```

2. **Create and Activate a Virtual Environment**:
   - On Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - On macOS/Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Generate Synthetic Data (Parquet)
Run the orchestrator script to generate 100K customers, 10K products, and 1M orders saved directly as compressed Parquet files into `data/raw/`:

```bash
python main.py
```

### 2. Execute PySpark Analytics Pipeline
Run the PySpark analysis runner to ingest Parquet files, compute business metrics, display reports with `.show()`, benchmark query times, and export outputs:

```bash
python run_analytics.py
```

### 3. Run Test Suite
Run unit tests with `pytest`:

```bash
pytest -v
```

---

## Output Datasets

| Dataset Path | Description |
| :--- | :--- |
| `data/raw/customers.csv` | Customer demographics, emails, locations, registration dates |
| `data/raw/products.csv` | Catalog with categories, unit prices, stock counts |
| `data/raw/orders.csv` | Transactions with quantities, totals, statuses, payment methods |
| `data/processed/category_performance/` | Aggregated revenue and order volumes by product category |
| `data/processed/top_customers/` | Top 10 customers ranked by lifetime completed spend |
| `data/processed/monthly_trends/` | Month-over-month revenue and active customer counts |
| `data/processed/order_status_distribution/` | Percentage breakdown of completed, cancelled, returned orders |
| `data/processed/top_products/` | Best-selling products ranked by gross revenue |

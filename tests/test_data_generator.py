"""Unit tests for the SyntheticDataGenerator class."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_generator import SyntheticDataGenerator


@pytest.fixture
def generator(tmp_path: Path) -> SyntheticDataGenerator:
    """Fixture providing a deterministic SyntheticDataGenerator instance with temporary output."""
    return SyntheticDataGenerator(seed=42, output_dir=tmp_path)


def test_generate_customers(generator: SyntheticDataGenerator) -> None:
    """Verifies customer generation schema, count, and age distribution properties."""
    num_customers: int = 500
    df: pd.DataFrame = generator.generate_customers(num_customers=num_customers)

    assert len(df) == num_customers
    expected_cols = {
        "customer_id",
        "name",
        "email",
        "age",
        "city",
        "country",
        "registration_date",
    }
    assert expected_cols.issubset(df.columns)
    assert df["customer_id"].iloc[0] == "CUST_000001"
    assert (df["age"] >= 18).all()
    assert (df["age"] <= 85).all()
    # Age mean should be approximately 35
    assert 30 <= df["age"].mean() <= 40


def test_generate_products(generator: SyntheticDataGenerator) -> None:
    """Verifies product generation schema, categories, and numerical bounds."""
    num_products: int = 100
    df: pd.DataFrame = generator.generate_products(num_products=num_products)

    assert len(df) == num_products
    expected_cols = {"product_id", "name", "category", "price", "stock", "rating"}
    assert expected_cols.issubset(df.columns)

    valid_categories = {"Electronics", "Clothing", "Home", "Sports", "Books"}
    assert set(df["category"].unique()).issubset(valid_categories)
    assert (df["price"] >= 10.0).all() and (df["price"] <= 500.0).all()
    assert (df["stock"] >= 0).all() and (df["stock"] <= 1000).all()
    assert (df["rating"] >= 1.0).all() and (df["rating"] <= 5.0).all()


def test_generate_orders_pareto_distribution(generator: SyntheticDataGenerator) -> None:
    """Validates order generation and verifies Pareto (80/20) skew."""
    customers_df = generator.generate_customers(num_customers=100)
    products_df = generator.generate_products(num_products=20)

    num_orders: int = 2000
    orders_df = generator.generate_orders(
        customers_df=customers_df,
        products_df=products_df,
        num_orders=num_orders,
        pareto_alpha=1.16,
    )

    assert len(orders_df) == num_orders
    expected_cols = {"order_id", "customer_id", "product_id", "quantity", "order_date"}
    assert expected_cols.issubset(orders_df.columns)

    # Referential integrity check
    assert orders_df["customer_id"].isin(customers_df["customer_id"]).all()
    assert orders_df["product_id"].isin(products_df["product_id"]).all()
    assert (orders_df["quantity"] >= 1).all() and (orders_df["quantity"] <= 10).all()

    # Check Pareto skew: top 20% of customers should hold the vast majority of orders
    order_counts = orders_df["customer_id"].value_counts()
    top_20_pct_count = int(len(customers_df) * 0.20)
    orders_from_top_20 = order_counts.iloc[:top_20_pct_count].sum()
    share = orders_from_top_20 / num_orders

    # Should be approximately 70-90% for a sample with alpha=1.16
    assert share >= 0.65, f"Expected Pareto skew >= 65%, got {share * 100:.2f}%"


def test_save_dataframe_to_csv(generator: SyntheticDataGenerator, tmp_path: Path) -> None:
    """Verifies saving DataFrame to CSV."""
    df = pd.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
    file_path = tmp_path / "sample.csv"
    generator.save_dataframe_to_csv(df, file_path)

    assert file_path.exists()
    loaded_df = pd.read_csv(file_path)
    assert len(loaded_df) == 2
    assert list(loaded_df.columns) == ["col_a", "col_b"]

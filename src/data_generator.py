"""High-performance synthetic e-commerce data generator.

Generates large-scale, relational datasets for customers, products, and orders
using Faker, NumPy statistical distributions, and Pandas.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from faker import Faker
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    CUSTOMERS_CSV,
    ORDERS_CSV,
    PRODUCTS_CSV,
    RAW_DATA_DIR,
    setup_logging,
)

logger: logging.Logger = setup_logging(name="data_generator")


class SyntheticDataGenerator:
    """Generates synthetic e-commerce datasets with realistic statistical distributions.

    Features:
        - Customer ages modeled via a Normal distribution around age 35.
        - Order frequencies modeled via a Pareto distribution (80/20 rule).
        - Product categories, ratings, and prices within defined constraints.
        - Progress bars via tqdm and execution logging throughout.
    """

    CATEGORIES: Sequence[str] = (
        "Electronics",
        "Clothing",
        "Home",
        "Sports",
        "Books",
    )

    def __init__(
        self,
        seed: int = 42,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Initializes the data generator with reproducible seeds and paths.

        Args:
            seed: Random seed for Faker and NumPy to ensure reproducibility.
            output_dir: Optional directory Path where generated CSVs will be saved.
        """
        self.seed: int = seed
        self.output_dir: Path = output_dir or RAW_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing SyntheticDataGenerator with seed=%d", seed)
        self.fake: Faker = Faker()
        Faker.seed(seed)
        np.random.seed(seed)

    def generate_customers(
        self,
        num_customers: int = 100_000
    ) -> pd.DataFrame:
        """Generates synthetic customer demographic and registration data.

        Age is sampled from a normal distribution centered around 35 (loc=35, scale=10)
        and clipped between 18 and 85.

        Args:
            num_customers: Total number of customer records to produce (default: 100,000).

        Returns:
            A pandas DataFrame containing customer profiles.
        """
        logger.info("Generating %d customers...", num_customers)

        # 1. Vectorized customer IDs
        customer_ids: List[str] = [f"CUST_{i:06d}" for i in range(1, num_customers + 1)]

        # 2. Normal distribution for age around 35 years
        raw_ages: np.ndarray = np.random.normal(loc=35.0, scale=10.0, size=num_customers)
        ages: np.ndarray = np.clip(raw_ages, 18, 85).astype(int)

        # 3. Seeded pool of geographical locations for speed & realistic distribution
        logger.debug("Generating geo-location pools...")
        cities_pool: List[str] = [self.fake.city() for _ in range(500)]
        countries_pool: List[str] = [
            "United States",
            "Canada",
            "United Kingdom",
            "Germany",
            "France",
            "Australia",
            "India",
            "Japan",
        ]
        cities: np.ndarray = np.random.choice(cities_pool, size=num_customers)
        countries: np.ndarray = np.random.choice(countries_pool, size=num_customers)

        # 4. Generate registration dates between 2021-01-01 and 2024-12-31
        start_epoch: int = int(datetime(2021, 1, 1).timestamp())
        end_epoch: int = int(datetime(2024, 12, 31).timestamp())
        random_timestamps: np.ndarray = np.random.randint(
            start_epoch, end_epoch, size=num_customers, dtype=np.int64
        )
        registration_dates: List[str] = [
            datetime.fromtimestamp(ts).strftime("%Y-%m-%d") for ts in random_timestamps
        ]

        # 5. Names and emails using Faker with progress tracking
        names: List[str] = []
        emails: List[str] = []
        domains: Sequence[str] = ("gmail.com", "yahoo.com", "outlook.com", "icloud.com", "example.com")

        logger.info("Synthesizing names and emails via Faker...")
        batch_size: int = 5_000
        with tqdm(total=num_customers, desc="Generating Customers", unit="cust") as pbar:
            for start_idx in range(0, num_customers, batch_size):
                end_idx: int = min(start_idx + batch_size, num_customers)
                for idx in range(start_idx, end_idx):
                    first: str = self.fake.first_name()
                    last: str = self.fake.last_name()
                    full_name: str = f"{first} {last}"
                    domain: str = domains[idx % len(domains)]
                    email: str = f"{first.lower()}.{last.lower()}.{idx + 1}@{domain}"

                    names.append(full_name)
                    emails.append(email)
                pbar.update(end_idx - start_idx)

        customers_df = pd.DataFrame({
            "customer_id": customer_ids,
            "name": names,
            "email": emails,
            "age": ages,
            "city": cities,
            "country": countries,
            "registration_date": registration_dates,
        })

        logger.info("Successfully generated %d customers.", len(customers_df))
        return customers_df

    def generate_products(
        self,
        num_products: int = 10_000
    ) -> pd.DataFrame:
        """Generates synthetic product catalog data with prices, stocks, and ratings.

        Args:
            num_products: Total number of products to produce (default: 10,000).

        Returns:
            A pandas DataFrame containing product catalog items.
        """
        logger.info("Generating %d products...", num_products)

        product_ids: List[str] = [f"PROD_{i:05d}" for i in range(1, num_products + 1)]

        # Category selection
        categories: np.ndarray = np.random.choice(self.CATEGORIES, size=num_products)

        # Price uniformly distributed between 10.0 and 500.0
        prices: np.ndarray = np.round(np.random.uniform(10.0, 500.0, size=num_products), 2)

        # Inventory stock levels between 0 and 1,000 units
        stocks: np.ndarray = np.random.randint(0, 1001, size=num_products)

        # Product ratings between 1.0 and 5.0 rounded to 1 decimal place
        ratings: np.ndarray = np.round(np.random.uniform(1.0, 5.0, size=num_products), 1)

        # Product names by category with progress tracking
        adjectives: Sequence[str] = ("Pro", "Ultra", "Classic", "Premium", "Elite", "Standard", "Eco")
        names: List[str] = []

        logger.info("Generating catalog product names...")
        with tqdm(total=num_products, desc="Generating Products", unit="prod") as pbar:
            for idx in range(num_products):
                cat: str = categories[idx]
                adj: str = adjectives[idx % len(adjectives)]
                noun: str = self.fake.word().capitalize()
                names.append(f"{adj} {cat} {noun} #{idx + 1}")
                pbar.update(1)

        products_df = pd.DataFrame({
            "product_id": product_ids,
            "name": names,
            "category": categories,
            "price": prices,
            "stock": stocks,
            "rating": ratings,
        })

        logger.info("Successfully generated %d products.", len(products_df))
        return products_df

    def generate_orders(
        self,
        customers_df: pd.DataFrame,
        products_df: pd.DataFrame,
        num_orders: int = 1_000_000,
        pareto_alpha: float = 1.16,
    ) -> pd.DataFrame:
        """Generates synthetic order transactions following a Pareto distribution.

        The Pareto shape parameter alpha ~ 1.16 models the 80/20 rule, ensuring
        that approximately 20% of customers generate 80% of all purchase transactions.

        Args:
            customers_df: Existing customers DataFrame.
            products_df: Existing products DataFrame.
            num_orders: Total number of orders to generate (default: 1,000,000).
            pareto_alpha: Shape parameter for customer order distribution (default: 1.16).

        Returns:
            A pandas DataFrame containing orders data.
        """
        logger.info(
            "Generating %d orders with Pareto distribution (alpha=%.2f)...",
            num_orders,
            pareto_alpha,
        )

        num_customers: int = len(customers_df)
        if num_customers == 0 or len(products_df) == 0:
            raise ValueError("Customers and products DataFrames must not be empty.")

        # 1. Compute Pareto sampling probabilities for customers
        logger.debug("Calculating Pareto weights across %d customers...", num_customers)
        raw_pareto_weights: np.ndarray = np.random.pareto(a=pareto_alpha, size=num_customers) + 1.0
        customer_probabilities: np.ndarray = raw_pareto_weights / raw_pareto_weights.sum()

        customer_id_pool: np.ndarray = customers_df["customer_id"].to_numpy()
        product_id_pool: np.ndarray = products_df["product_id"].to_numpy()

        # 2. Date ranges for transactions (2023-01-01 to 2025-06-30)
        start_epoch: int = int(datetime(2023, 1, 1).timestamp())
        end_epoch: int = int(datetime(2025, 6, 30).timestamp())

        # 3. Process orders in memory-efficient batches with tqdm progress bar
        batch_size: int = 250_000
        num_batches: int = (num_orders + batch_size - 1) // batch_size

        order_id_list: List[str] = []
        cust_id_list: List[str] = []
        prod_id_list: List[str] = []
        qty_list: List[int] = []
        date_list: List[str] = []

        logger.info("Synthesizing order records in %d batches...", num_batches)
        with tqdm(total=num_orders, desc="Generating Orders", unit="orders") as pbar:
            for batch_idx in range(num_batches):
                current_batch_size: int = min(batch_size, num_orders - (batch_idx * batch_size))
                offset: int = batch_idx * batch_size

                # Order IDs
                batch_order_ids: List[str] = [
                    f"ORD_{offset + i:07d}" for i in range(1, current_batch_size + 1)
                ]

                # Pareto-weighted customer sampling
                sampled_customers: np.ndarray = np.random.choice(
                    customer_id_pool,
                    size=current_batch_size,
                    p=customer_probabilities,
                )

                # Uniform product sampling
                sampled_products: np.ndarray = np.random.choice(
                    product_id_pool,
                    size=current_batch_size,
                )

                # Quantities between 1 and 10
                quantities: np.ndarray = np.random.randint(1, 11, size=current_batch_size)

                # Order dates
                random_ts: np.ndarray = np.random.randint(
                    start_epoch, end_epoch, size=current_batch_size, dtype=np.int64
                )
                order_dates: List[str] = [
                    datetime.fromtimestamp(ts).strftime("%Y-%m-%d") for ts in random_ts
                ]

                order_id_list.extend(batch_order_ids)
                cust_id_list.extend(sampled_customers)
                prod_id_list.extend(sampled_products)
                qty_list.extend(quantities.tolist())
                date_list.extend(order_dates)

                pbar.update(current_batch_size)

        orders_df = pd.DataFrame({
            "order_id": order_id_list,
            "customer_id": cust_id_list,
            "product_id": prod_id_list,
            "quantity": qty_list,
            "order_date": date_list,
        })

        logger.info("Successfully generated %d orders.", len(orders_df))
        return orders_df

    def save_dataframe_to_csv(
        self,
        df: pd.DataFrame,
        file_path: Path
    ) -> None:
        """Saves a pandas DataFrame to CSV format.

        Args:
            df: The DataFrame to write.
            file_path: Target Path on disk.
        """
        logger.info("Writing %d rows to '%s'...", len(df), file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False, encoding="utf-8")
        logger.info("Saved '%s'.", file_path.name)

    def generate_all(
        self,
        num_customers: int = 100_000,
        num_products: int = 10_000,
        num_orders: int = 1_000_000,
        save_to_disk: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Runs end-to-end generation for customers, products, and orders.

        Args:
            num_customers: Number of customers (default: 100,000).
            num_products: Number of products (default: 10,000).
            num_orders: Number of orders (default: 1,000,000).
            save_to_disk: If True, writes datasets to output_dir as CSVs.

        Returns:
            A tuple of (customers_df, products_df, orders_df).
        """
        logger.info("Starting synthetic e-commerce data generation pipeline...")

        customers_df = self.generate_customers(num_customers=num_customers)
        products_df = self.generate_products(num_products=num_products)
        orders_df = self.generate_orders(
            customers_df=customers_df,
            products_df=products_df,
            num_orders=num_orders,
        )

        if save_to_disk:
            self.save_dataframe_to_csv(customers_df, self.output_dir / "customers.csv")
            self.save_dataframe_to_csv(products_df, self.output_dir / "products.csv")
            self.save_dataframe_to_csv(orders_df, self.output_dir / "orders.csv")

        logger.info("Pipeline generation complete!")
        return customers_df, products_df, orders_df


if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    generator.generate_all(
        num_customers=100_000,
        num_products=10_000,
        num_orders=1_000_000,
        save_to_disk=True,
    )

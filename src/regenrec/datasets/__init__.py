"""Adapters for public recommendation datasets."""

from regenrec.datasets.amazon2023 import (
    AMAZON2023_DATASET_ID,
    AmazonPreparationStats,
    download_amazon2023,
    iter_amazon_examples,
    prepare_amazon2023,
)
from regenrec.datasets.recif import (
    RECIF_DATASET_ID,
    RECIF_PRODUCT_BEHAVIOR,
    RecIFPreparationStats,
    build_catalog_from_sid_rows,
    iter_product_examples,
    prepare_recif_product,
)

__all__ = [
    "AMAZON2023_DATASET_ID",
    "AmazonPreparationStats",
    "RECIF_DATASET_ID",
    "RECIF_PRODUCT_BEHAVIOR",
    "RecIFPreparationStats",
    "build_catalog_from_sid_rows",
    "download_amazon2023",
    "iter_amazon_examples",
    "iter_product_examples",
    "prepare_amazon2023",
    "prepare_recif_product",
]

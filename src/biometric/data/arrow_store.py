"""PyArrow-based data catalog for efficient metadata storage and querying.

The ArrowBiometricStore provides a Parquet-backed catalog of all biometric
images with their metadata and quality metrics. This replaces expensive
filesystem walks with single-file columnar scans and enables SQL-like
filtering (e.g., "all left iris images for male subjects with blur_score > 100").

Benefits over raw filesystem traversal:
- Single file scan vs. os.walk across 135+ directories.
- Columnar pruning: read only the columns you need.
- Predicate pushdown: filter rows at the I/O level.
- Language-agnostic: Parquet is readable from Python, Spark, DuckDB, etc.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Schema definition for the biometric catalog.
# Explicit schema ensures type safety and consistent serialization.
CATALOG_SCHEMA = pa.schema([
    pa.field("subject_id", pa.int32(), nullable=False),
    pa.field("gender", pa.string(), nullable=False),
    pa.field("modality", pa.string(), nullable=False),     # 'fingerprint' | 'iris'
    pa.field("side", pa.string(), nullable=False),          # 'left' | 'right'
    pa.field("detail", pa.string(), nullable=False),        # finger type or 'eye'
    pa.field("sample_idx", pa.int32(), nullable=False),
    pa.field("image_path", pa.string(), nullable=False),
    pa.field("width", pa.int32(), nullable=False),
    pa.field("height", pa.int32(), nullable=False),
    pa.field("brightness_mean", pa.float32(), nullable=False),
    pa.field("contrast_std", pa.float32(), nullable=False),
    pa.field("blur_score", pa.float32(), nullable=False),
])


class ArrowBiometricStore:
    """Read/write interface for the Parquet-backed biometric catalog.

    Example:
        store = ArrowBiometricStore()
        store.write(records, Path("catalog.parquet"))

        table = store.read(Path("catalog.parquet"))
        iris_only = store.read(
            Path("catalog.parquet"),
            filters=[("modality", "=", "iris")],
        )
    """

    def write(self, records: list[dict[str, Any]], path: Path) -> None:
        """Write a list of record dicts to a Parquet file.

        Args:
            records: List of dicts matching CATALOG_SCHEMA field names.
            path: Output Parquet file path.
        """
        if not records:
            logger.warning("No records to write.")
            return

        # Build column arrays from records
        arrays = []
        for field in CATALOG_SCHEMA:
            values = [record[field.name] for record in records]
            arrays.append(pa.array(values, type=field.type))

        table = pa.table(arrays, schema=CATALOG_SCHEMA)

        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            table,
            str(path),
            compression="snappy",
            write_statistics=True,
        )
        logger.info(
            "Wrote %d records (%d columns) to %s (%.1f KB)",
            table.num_rows,
            table.num_columns,
            path,
            path.stat().st_size / 1024,
        )

    def read(
        self,
        path: Path,
        columns: list[str] | None = None,
        filters: list[tuple[str, str, Any]] | None = None,
    ) -> pa.Table:
        """Read the catalog from a Parquet file with optional filtering.

        Args:
            path: Path to the Parquet file.
            columns: Optional list of column names to read (columnar pruning).
            filters: Optional list of (column, op, value) filter tuples
                     for predicate pushdown. E.g., [("modality", "=", "iris")].

        Returns:
            PyArrow Table with the requested data.
        """
        table = pq.read_table(
            str(path),
            columns=columns,
            filters=filters,
        )
        logger.info("Read %d records from %s", table.num_rows, path)
        return table

    def read_pandas(self, path: Path, **kwargs: Any) -> Any:
        """Read the catalog as a pandas DataFrame.

        Args:
            path: Path to the Parquet file.
            **kwargs: Passed to self.read().

        Returns:
            pandas DataFrame.
        """
        table = self.read(path, **kwargs)
        return table.to_pandas()

    def summary(self, path: Path) -> dict[str, Any]:
        """Generate a summary report of the catalog contents.

        Args:
            path: Path to the Parquet file.

        Returns:
            Dict with counts, modality breakdown, and quality stats.
        """
        table = self.read(path)

        modality_counts = {}
        for modality in ["fingerprint", "iris"]:
            mask = pa.compute.equal(table.column("modality"), modality)
            modality_counts[modality] = pa.compute.sum(mask.cast(pa.int64())).as_py()

        # Count unique subjects per gender using pure PyArrow (no pandas dependency)
        subject_ids = table.column("subject_id").to_pylist()
        gender_list = table.column("gender").to_pylist()
        subject_gender = {sid: g for sid, g in zip(subject_ids, gender_list)}
        unique_subjects = set(subject_gender.keys())

        gender_counts = {}
        for gender in ["M", "F"]:
            gender_counts[gender] = sum(
                1 for sid in unique_subjects if subject_gender[sid] == gender
            )

        brightness = table.column("brightness_mean")
        blur = table.column("blur_score")

        return {
            "total_records": table.num_rows,
            "modality_counts": modality_counts,
            "gender_counts": gender_counts,
            "num_subjects": len(unique_subjects),
            "brightness_mean_avg": float(pa.compute.mean(brightness).as_py()),
            "blur_score_avg": float(pa.compute.mean(blur).as_py()),
        }

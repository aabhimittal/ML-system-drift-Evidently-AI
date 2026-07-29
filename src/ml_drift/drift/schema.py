"""Data-contract / schema validation — the first line of defence in production.

Before you even ask "has the distribution drifted?", you must ask "is this batch
even the shape I expect?". In real pipelines the nastiest incidents are not
subtle statistical drift but hard schema breaks: a renamed column, a field that
started arriving as a string, a null-rate that jumped from 0% to 40%, an
out-of-range value from an upstream bug, or a brand-new categorical level.

:class:`SchemaContract` is inferred from the reference (training) data and then
enforced against each incoming batch. :func:`validate_batch` returns a structured
:class:`SchemaReport` listing every violation, classified by severity.

Severity
--------
* ``error``   — breaks scoring or silently corrupts it (missing column, dtype
  change on a required column, values outside the learned range by a wide margin).
* ``warning`` — tolerable but worth alerting (new column, unseen category, a
  null-rate spike, a moderate range excursion).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ColumnContract:
    name: str
    dtype_kind: str            # numpy dtype kind: 'i','f','O','b', etc.
    is_numeric: bool
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    max_null_rate: float = 0.0
    allowed_categories: Optional[List[str]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype_kind": self.dtype_kind,
            "is_numeric": self.is_numeric,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "max_null_rate": round(self.max_null_rate, 4),
            "allowed_categories": self.allowed_categories,
        }


@dataclass
class SchemaViolation:
    column: str
    kind: str        # missing_column | unexpected_column | dtype_change |
                     # null_spike | out_of_range | unseen_category
    severity: str    # error | warning
    detail: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "kind": self.kind,
            "severity": self.severity,
            "detail": self.detail,
        }


@dataclass
class SchemaReport:
    ok: bool
    n_errors: int
    n_warnings: int
    violations: List[SchemaViolation] = field(default_factory=list)

    @property
    def errors(self) -> List[SchemaViolation]:
        return [v for v in self.violations if v.severity == "error"]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "n_errors": self.n_errors,
            "n_warnings": self.n_warnings,
            "violations": [v.as_dict() for v in self.violations],
        }


@dataclass
class SchemaContract:
    """A data contract inferred from a reference frame and enforced on batches."""

    columns: Dict[str, ColumnContract] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)

    @classmethod
    def infer(
        cls,
        reference: pd.DataFrame,
        ignore: Optional[List[str]] = None,
        range_slack: float = 0.0,
        null_slack: float = 0.02,
        max_categories: int = 50,
    ) -> "SchemaContract":
        """Infer a contract from ``reference``.

        Parameters
        ----------
        range_slack: fraction of the reference range to pad min/max by, so a
            slightly-wider-but-plausible value is not an error.
        null_slack: how much above the reference null rate is tolerated before a
            null-spike warning fires.
        max_categories: skip the allowed-category set for very high-cardinality
            columns (treat them as free-form; only null/dtype checks apply).
        """
        ignore = set(ignore or [])
        columns: Dict[str, ColumnContract] = {}
        required: List[str] = []
        for name in reference.columns:
            if name in ignore:
                continue
            col = reference[name]
            is_num = pd.api.types.is_numeric_dtype(col)
            null_rate = float(col.isna().mean())
            contract = ColumnContract(
                name=name,
                dtype_kind=col.dtype.kind,
                is_numeric=is_num,
                max_null_rate=min(1.0, null_rate + null_slack),
            )
            if is_num:
                finite = col.to_numpy(dtype=float)
                finite = finite[np.isfinite(finite)]
                if finite.size:
                    lo, hi = float(finite.min()), float(finite.max())
                    pad = (hi - lo) * range_slack
                    contract.min_value = lo - pad
                    contract.max_value = hi + pad
            else:
                cats = col.dropna().astype(str).unique().tolist()
                if len(cats) <= max_categories:
                    contract.allowed_categories = sorted(cats)
            columns[name] = contract
            required.append(name)
        return cls(columns=columns, required=required)

    def validate(
        self,
        batch: pd.DataFrame,
        out_of_range_error_frac: float = 0.05,
    ) -> SchemaReport:
        """Validate ``batch`` against this contract.

        ``out_of_range_error_frac``: if more than this fraction of a numeric
        column's values fall outside the learned range, it is escalated to an
        error (a systematic upstream problem) rather than a warning (a few outliers).
        """
        violations: List[SchemaViolation] = []

        # Missing required columns (error) and unexpected new columns (warning).
        for name in self.required:
            if name not in batch.columns:
                violations.append(SchemaViolation(
                    name, "missing_column", "error",
                    f"required column '{name}' is absent from the batch"))
        for name in batch.columns:
            if name not in self.columns:
                violations.append(SchemaViolation(
                    name, "unexpected_column", "warning",
                    f"column '{name}' is not part of the contract"))

        # Per-column checks for the columns we do have.
        for name, contract in self.columns.items():
            if name not in batch.columns:
                continue
            col = batch[name]

            # Null-rate spike.
            null_rate = float(col.isna().mean())
            if null_rate > contract.max_null_rate + 1e-9:
                violations.append(SchemaViolation(
                    name, "null_spike", "warning",
                    f"null rate {null_rate:.2%} exceeds contract "
                    f"{contract.max_null_rate:.2%}"))

            if contract.is_numeric:
                if not pd.api.types.is_numeric_dtype(col):
                    coerced = pd.to_numeric(col, errors="coerce")
                    # If coercion turns most values into NaN, the type truly changed.
                    lost = float(coerced.isna().mean() - col.isna().mean())
                    sev = "error" if lost > 0.5 else "warning"
                    violations.append(SchemaViolation(
                        name, "dtype_change", sev,
                        f"expected numeric, got {col.dtype}; "
                        f"{lost:.0%} of values un-parseable"))
                    col = coerced
                if contract.min_value is not None and contract.max_value is not None:
                    vals = col.to_numpy(dtype=float)
                    finite = vals[np.isfinite(vals)]
                    if finite.size:
                        oob = np.mean((finite < contract.min_value)
                                      | (finite > contract.max_value))
                        if oob > 0:
                            sev = "error" if oob > out_of_range_error_frac else "warning"
                            violations.append(SchemaViolation(
                                name, "out_of_range", sev,
                                f"{oob:.2%} of values outside learned range "
                                f"[{contract.min_value:.4g}, {contract.max_value:.4g}]"))
            else:
                # Categorical dtype change (numeric arriving where text expected).
                if pd.api.types.is_numeric_dtype(col):
                    violations.append(SchemaViolation(
                        name, "dtype_change", "warning",
                        f"expected categorical/text, got numeric {col.dtype}"))
                if contract.allowed_categories is not None:
                    seen = set(col.dropna().astype(str).unique())
                    unseen = sorted(seen - set(contract.allowed_categories))
                    if unseen:
                        preview = ", ".join(unseen[:5]) + ("..." if len(unseen) > 5 else "")
                        violations.append(SchemaViolation(
                            name, "unseen_category", "warning",
                            f"{len(unseen)} new categ/ies not seen in training: {preview}"))

        n_err = sum(1 for v in violations if v.severity == "error")
        n_warn = sum(1 for v in violations if v.severity == "warning")
        return SchemaReport(ok=(n_err == 0), n_errors=n_err, n_warnings=n_warn,
                            violations=violations)


def validate_batch(reference: pd.DataFrame, batch: pd.DataFrame,
                    ignore: Optional[List[str]] = None, **infer_kwargs) -> SchemaReport:
    """Convenience: infer a contract from ``reference`` and validate ``batch``."""
    contract = SchemaContract.infer(reference, ignore=ignore, **infer_kwargs)
    return contract.validate(batch)

"""
Canonical country list -- the 12-way classification target for the hierarchical
country -> within-country-cell architecture (`src/models/hierarchical_geo_classifier.py`).

Fixed, alphabetically sorted, matching `train_labels.csv`'s exact 12 values (CHALLENGE.md's
country list; multi-word names are underscore-joined, e.g. "United_Kingdom"). Hardcoded as
a literal list here rather than re-derived via `sorted(df.country.unique())` at import time
-- so the index<->country mapping can never silently shift if some future caller passes a
filtered/partial dataframe that happens not to contain all 12 countries.
"""

from __future__ import annotations

COUNTRIES = [
    "Belarus",
    "Finland",
    "France",
    "Germany",
    "Iceland",
    "Italy",
    "Norway",
    "Poland",
    "Spain",
    "Sweden",
    "Turkey",
    "United_Kingdom",
]
COUNTRY_TO_IDX = {c: i for i, c in enumerate(COUNTRIES)}
N_COUNTRIES = len(COUNTRIES)

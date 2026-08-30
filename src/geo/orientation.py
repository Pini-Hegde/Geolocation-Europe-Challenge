"""
Orientation correction from the challenge-provided `geo_dataset/orientation_flags.json`.

That file has one boolean per image, for BOTH the training set and the holdout set
(14,158 keys = 11,758 train + 2,400 holdout). ~8.9% are `true`. Inspecting the flagged
images shows they are rotated 180 degrees (upside down: dashboard/car roof at the top,
sky in the lower half). The rest of this project's pipeline historically ignored this
file entirely.

`true`  -> the image is upside down; rotate it 180 degrees before use.
`false` -> leave as-is.

Applied identically at train and inference time (opt-in via a `fix_orientation` flag on
`GeoDataset` / the holdout dataset) so the model always sees upright scenes. This is a
lossless transform of the provided image, not external or synthetic data.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
ORIENTATION_FLAGS_JSON = REPO_ROOT / "geo_dataset" / "orientation_flags.json"


@lru_cache(maxsize=1)
def load_orientation_flags() -> dict[str, bool]:
    """filename (with .jpg) -> True if the image is upside down and needs a 180 rotation."""
    if not ORIENTATION_FLAGS_JSON.exists():
        raise FileNotFoundError(
            f"{ORIENTATION_FLAGS_JSON} not found -- needed for orientation correction. "
            "It ships with geo_dataset/."
        )
    return json.loads(ORIENTATION_FLAGS_JSON.read_text())


def fix_orientation(image: Image.Image, filename: str) -> Image.Image:
    """Rotate `image` 180 degrees iff `filename` is flagged upside down. Unknown filenames
    (not in the flags file) are left untouched."""
    if load_orientation_flags().get(filename, False):
        return image.transpose(Image.Transpose.ROTATE_180)
    return image

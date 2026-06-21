"""Feature extraction used by the final imitation-learning policy.

The policy uses four compact, interpretable features:

1. x-coordinate of the largest yellow blob centroid
2. y-coordinate of the largest yellow blob centroid
3. left/right yellow-pixel skew
4. fraction of yellow pixels in the image
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.ndimage import label


class _RngLike(Protocol):
    def randint(self, low: int, high: int | None = None, size=None): ...

    def integers(self, low: int, high: int | None = None, size=None): ...


@dataclass(frozen=True)
class YellowMaskConfig:
    """RGB thresholds for detecting yellow pickup objects."""

    r_min: int = 200
    g_min: int = 200
    b_max: int = 50


def _as_batch(observations: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(observations)
    if arr.ndim == 3:
        return arr[None, ...], True
    if arr.ndim == 4:
        return arr, False
    raise ValueError(
        "Expected one RGB observation with shape (H, W, 3) or a batch "
        f"with shape (N, H, W, 3), got {arr.shape}."
    )


def _randint(rng: _RngLike | None, low: int, high: int) -> int:
    if rng is None:
        return int(np.random.randint(low, high))
    if hasattr(rng, "integers"):
        return int(rng.integers(low, high))
    return int(rng.randint(low, high))


def yellow_mask_rgb(
    observations: np.ndarray,
    config: YellowMaskConfig = YellowMaskConfig(),
) -> np.ndarray:
    """Return a boolean mask for yellow pixels.

    The thresholds are tuned for the yellow target objects.
    """

    batch, single = _as_batch(observations)
    red = batch[..., 0]
    green = batch[..., 1]
    blue = batch[..., 2]
    mask = (red >= config.r_min) & (green >= config.g_min) & (blue <= config.b_max)
    return mask[0] if single else mask


def yellow_quantity_and_skew(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute left/right yellow skew and yellow-pixel fraction."""

    mask_batch = mask[None, ...] if mask.ndim == 2 else np.asarray(mask)
    if mask_batch.ndim != 3:
        raise ValueError(f"Expected mask shape (H, W) or (N, H, W), got {mask.shape}.")

    n_samples, height, width = mask_batch.shape
    quantity = np.empty(n_samples, dtype=np.float32)
    skew = np.empty(n_samples, dtype=np.float32)

    for i in range(n_samples):
        count = float(mask_batch[i].sum())
        left = float(mask_batch[i, :, : width // 2].sum())
        right = float(mask_batch[i, :, width // 2 :].sum())
        quantity[i] = count / float(height * width)
        skew[i] = (left - right) / (count + 1e-6)

    return skew, quantity


def largest_yellow_centroid(
    mask: np.ndarray,
    rng: _RngLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return centroid coordinates for the largest yellow connected component.

    If no yellow component is visible, inject a random centroid. This behavior
    keeps the policy searching instead of locking onto a fixed no-target action.
    """

    mask_batch = mask[None, ...] if mask.ndim == 2 else np.asarray(mask)
    if mask_batch.ndim != 3:
        raise ValueError(f"Expected mask shape (H, W) or (N, H, W), got {mask.shape}.")

    n_samples, height, width = mask_batch.shape
    xs_mean = np.empty(n_samples, dtype=np.float32)
    ys_mean = np.empty(n_samples, dtype=np.float32)

    for i in range(n_samples):
        labels, n_components = label(mask_batch[i])
        if n_components == 0:
            xs_mean[i] = _randint(rng, 0, width)
            ys_mean[i] = _randint(rng, 0, height)
            continue

        sizes = np.bincount(labels.ravel())[1:]
        largest_component = int(sizes.argmax() + 1)
        ys, xs = np.nonzero(labels == largest_component)
        xs_mean[i] = xs.mean()
        ys_mean[i] = ys.mean()

    return xs_mean, ys_mean


def extract_features(
    observations: np.ndarray,
    config: YellowMaskConfig = YellowMaskConfig(),
    rng: _RngLike | None = None,
) -> np.ndarray:
    """Extract the final four-feature representation from RGB observations."""

    batch, _ = _as_batch(observations)
    mask = yellow_mask_rgb(batch, config=config)
    features = np.zeros((batch.shape[0], 4), dtype=np.float32)

    features[:, 0], features[:, 1] = largest_yellow_centroid(mask, rng=rng)
    features[:, 2], features[:, 3] = yellow_quantity_and_skew(mask)
    return features


def extract_single_features(
    observation: np.ndarray,
    config: YellowMaskConfig = YellowMaskConfig(),
    rng: _RngLike | None = None,
) -> np.ndarray:
    """Extract a one-dimensional feature vector for a single observation."""

    return extract_features(observation, config=config, rng=rng)[0]

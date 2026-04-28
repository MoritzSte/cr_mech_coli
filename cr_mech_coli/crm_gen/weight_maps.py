"""Soft foreground / background weight maps for region-aware metrics.

Replaces the previous hard-masking approach (zero out pixels outside the
region, then recompute SSIM on the masked image).  Hard masking creates
sharp zero-edges at every cell boundary that SSIM's sliding window then
treats as real image content — the score ends up partly measuring the
masking artifact instead of the image.

With soft weight maps, SSIM is computed once on the unmodified full image.
The per-pixel SSIM map is then aggregated twice — once weighted by the
foreground map, once by the background map.  No sliding window ever
crosses an artificial edge, and features that straddle the cell boundary
(halo, edge fringe) aren't split across two competing loss terms.
"""

from typing import Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt


def build_region_weight_maps(
    mask: np.ndarray, sigma_px: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Build soft foreground / background weight maps from a segmentation mask.

    Uses a Gaussian falloff based on distance from the nearest cell pixel::

        w_fg(x, y) = exp(-(d / sigma_px)^2)
        w_bg(x, y) = 1 - w_fg(x, y)

    where ``d`` is the Euclidean distance from ``(x, y)`` to the nearest
    foreground pixel.  ``w_fg`` is ~1 inside cells, falls off smoothly
    outward, and reaches ~0.37 at one ``sigma_px`` beyond the cell edge.

    Args:
        mask: 2-D or 3-D segmentation mask.  Any non-zero entry is treated
            as cell (foreground).  A 3-D mask is collapsed via ``np.max``
            over the last axis before the distance transform.
        sigma_px: Falloff scale in pixels.  Larger values extend the
            foreground's sphere of influence outward, capturing halo /
            fringe zones in the foreground term.  Typical range 20-60 px.

    Returns:
        ``(w_fg, w_bg)`` — two float arrays with the spatial shape of
        ``mask``, both in ``[0, 1]`` and summing to 1 pixel-wise.
    """
    if sigma_px <= 0:
        raise ValueError(f"sigma_px must be positive, got {sigma_px}")

    if mask.ndim == 3:
        mask = np.max(mask, axis=-1)

    is_fg = mask > 0
    if not is_fg.any():
        # No cells in this mask — everything is background.
        w_fg = np.zeros(mask.shape, dtype=np.float64)
        w_bg = np.ones(mask.shape, dtype=np.float64)
        return w_fg, w_bg

    d = distance_transform_edt(~is_fg)
    w_fg = np.exp(-((d / sigma_px) ** 2))
    w_bg = 1.0 - w_fg
    return w_fg, w_bg

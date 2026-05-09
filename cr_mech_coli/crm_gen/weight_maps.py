"""Soft foreground / background / edge-band weight maps for region-aware metrics.

Replaces the previous hard-masking approach (zero out pixels outside the
region, then recompute SSIM on the masked image).  Hard masking creates
sharp zero-edges at every cell boundary that SSIM's sliding window then
treats as real image content — the score ends up partly measuring the
masking artifact instead of the image.

With soft weight maps, SSIM is computed once on the unmodified full image.
The per-pixel SSIM map is then aggregated against each region's weight
map.  No sliding window ever crosses an artificial edge, and features
that straddle the cell boundary (halo, edge fringe) aren't split across
two competing loss terms.

The optional ``edge_sigma_px`` parameter adds a third weight map that
peaks on the cell boundary itself, with a narrow Gaussian falloff.
This gives small spatially-localised effects (edge fringe, halo
transition, absorption tail) dedicated signal in the loss, since global
SSIM / LPIPS / power-spectrum metrics under-weight features that occupy
only a thin pixel band.
"""

from typing import Optional, Tuple, Union

import numpy as np
from scipy.ndimage import distance_transform_edt


def build_region_weight_maps(
    mask: np.ndarray,
    sigma_px: float,
    edge_sigma_px: Optional[float] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Build soft foreground / background (and optionally edge-band) weight maps.

    Foreground / background use a Gaussian falloff based on distance from
    the nearest cell pixel::

        w_fg(x, y) = exp(-(d_out / sigma_px)^2)
        w_bg(x, y) = 1 - w_fg(x, y)

    where ``d_out`` is the Euclidean distance from ``(x, y)`` to the
    nearest foreground pixel.  ``w_fg`` is ~1 inside cells, falls off
    smoothly outward, and reaches ~0.37 at one ``sigma_px`` beyond the
    cell edge.

    When ``edge_sigma_px`` is supplied, a third weight map is also
    returned that peaks on the cell boundary::

        w_edge(x, y) = exp(-(d_to_edge / edge_sigma_px)^2)

    where ``d_to_edge`` is the distance from ``(x, y)`` to the nearest
    boundary pixel (treating the boundary as living half a pixel away
    from the boundary-adjacent layer).  Concretely we use
    ``d_to_edge = min(d_in, d_out) - 0.5``, which puts the peak inside
    the inner / outer one-pixel boundary band and decays to ~0.37 at
    one ``edge_sigma_px`` to either side.  Useful for small,
    spatially-localised features (edge fringe, halo transition) that
    are otherwise drowned out by global metrics.

    Args:
        mask: 2-D or 3-D segmentation mask.  Any non-zero entry is treated
            as cell (foreground).  A 3-D mask is collapsed via ``np.max``
            over the last axis before the distance transform.
        sigma_px: Foreground falloff scale in pixels.  Larger values
            extend the foreground's sphere of influence outward, capturing
            halo / fringe zones in the foreground term.  Typical
            range 20-60 px.
        edge_sigma_px: Optional edge-band falloff scale in pixels.  When
            ``None`` (default), only ``(w_fg, w_bg)`` are returned —
            preserving the original two-region API.  Typical range
            1-4 px (cell-boundary blur is usually ≈ 1.5 px).

    Returns:
        ``(w_fg, w_bg)`` if ``edge_sigma_px`` is None, otherwise
        ``(w_fg, w_bg, w_edge)``.  Each array has the spatial shape of
        ``mask`` and is in ``[0, 1]``.
    """
    if sigma_px <= 0:
        raise ValueError(f"sigma_px must be positive, got {sigma_px}")
    if edge_sigma_px is not None and edge_sigma_px <= 0:
        raise ValueError(
            f"edge_sigma_px must be positive when given, got {edge_sigma_px}"
        )

    if mask.ndim == 3:
        mask = np.max(mask, axis=-1)

    is_fg = mask > 0
    if not is_fg.any():
        # No cells in this mask — everything is background, edge band empty.
        w_fg = np.zeros(mask.shape, dtype=np.float64)
        w_bg = np.ones(mask.shape, dtype=np.float64)
        if edge_sigma_px is None:
            return w_fg, w_bg
        w_edge = np.zeros(mask.shape, dtype=np.float64)
        return w_fg, w_bg, w_edge

    d_out = distance_transform_edt(~is_fg)
    w_fg = np.exp(-((d_out / sigma_px) ** 2))
    w_bg = 1.0 - w_fg

    if edge_sigma_px is None:
        return w_fg, w_bg

    d_in = distance_transform_edt(is_fg)
    # ``distance_transform_edt`` returns 0 inside the "other" region —
    # so d_in is 0 for every background pixel and d_out is 0 for every
    # foreground pixel.  ``min(d_in, d_out)`` is therefore 0 everywhere
    # (useless).  Instead, pick the right field per pixel: foreground
    # pixels use d_in (distance going into the cell), background pixels
    # use d_out (distance going out from the cell).  The boundary lives
    # half a pixel inside the d=1 layer, so subtract 0.5 and clamp so
    # the boundary band itself gets the maximum weight.
    unsigned_d = np.where(is_fg, d_in, d_out)
    d_to_edge = np.maximum(unsigned_d - 0.5, 0.0)
    w_edge = np.exp(-((d_to_edge / edge_sigma_px) ** 2))
    return w_fg, w_bg, w_edge

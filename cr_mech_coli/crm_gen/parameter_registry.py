"""
Central registry of all tunable parameters for synthetic microscope image generation.

Every parameter that can be optimized or screened is defined here with its
metadata (bounds, type, default, category).  This replaces the hardcoded
parameter lists.

To add a new parameter (e.g. for a new imaging modality):
    1.  Add a ``ParameterDef`` entry to ``PARAMETER_REGISTRY``.
    2.  Wire it into ``scene.py:apply_synthetic_effects()`` via the
        ``params`` dict pathway.
    3.  Re-run ``crm_gen screen`` — the screening will automatically
        evaluate whether the new parameter matters.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class ParameterDef:
    """Definition of a single tunable parameter."""

    name: str
    bounds: Tuple[float, float]
    dtype: str  # "float" or "int"
    default: Any
    category: str  # "background", "halo", "psf", "noise", "brightness"
    description: str


# ---------------------------------------------------------------------------
# The registry — single source of truth for all optimizable parameters
# ---------------------------------------------------------------------------

PARAMETER_REGISTRY: Dict[str, ParameterDef] = {
    # ── Background ────────────────────────────────────────────────────────
    "bg_base_brightness": ParameterDef(
        name="bg_base_brightness",
        bounds=(0.0, 0.6),
        dtype="float",
        default=0.3,
        category="background",
        description="Base brightness level of the background",
    ),
    "bg_gradient_strength": ParameterDef(
        name="bg_gradient_strength",
        bounds=(0.0, 0.04),
        dtype="float",
        default=0.001,
        category="background",
        description="Illumination gradie1nt intensity",
    ),
    "bg_noise_scale": ParameterDef(
        name="bg_noise_scale",
        bounds=(1, 25),
        dtype="int",
        default=5,
        category="background",
        description="Perlin-like background noise scale factor",
    ),
    "texture_strength": ParameterDef(
        name="texture_strength",
        bounds=(0.0, 0.1),
        dtype="float",
        default=0.02,
        category="background",
        description="Fine texture strength for background surface variations",
    ),
    "texture_scale": ParameterDef(
        name="texture_scale",
        bounds=(0.5, 5.0),
        dtype="float",
        default=1.5,
        category="background",
        description="Texture smoothness (higher = smoother)",
    ),
    "bg_blur_sigma": ParameterDef(
        name="bg_blur_sigma",
        bounds=(0.1, 3.0),
        dtype="float",
        default=0.8,
        category="background",
        description="Gaussian blur sigma for background optical effects",
    ),
    "dark_spot_intensity": ParameterDef(
        name="dark_spot_intensity",
        bounds=(0.0, 0.3),
        dtype="float",
        default=0.15,
        category="background",
        description="Intensity (darkness) of debris spots",
    ),
    "num_dark_spots_max": ParameterDef(
        name="num_dark_spots_max",
        bounds=(0, 10),
        dtype="int",
        default=0,
        category="background",
        description="Maximum number of dark spots in background",
    ),
    # ── Halo ──────────────────────────────────────────────────────────────
    "bac_halo_intensity": ParameterDef(
        name="bac_halo_intensity",
        bounds=(0.0, 0.6),
        dtype="float",
        default=0.00,
        category="halo",
        description="Halo effect intensity around bacteria edges",
    ),
    "halo_inner_width": ParameterDef(
        name="halo_inner_width",
        bounds=(0.5, 5.0),
        dtype="float",
        default=2.0,
        category="halo",
        description="Width of inner bright halo in pixels",
    ),
    "halo_outer_width": ParameterDef(
        name="halo_outer_width",
        bounds=(5.0, 100.0),
        dtype="float",
        default=50.0,
        category="halo",
        description="Total halo extent in pixels",
    ),
    "halo_blur_sigma": ParameterDef(
        name="halo_blur_sigma",
        bounds=(0.1, 3.0),
        dtype="float",
        default=0.5,
        category="halo",
        description="Gaussian blur sigma for halo transition smoothing",
    ),
    # ── PSF ───────────────────────────────────────────────────────────────
    "psf_sigma": ParameterDef(
        name="psf_sigma",
        bounds=(0.1, 3.0),
        dtype="float",
        default=1.0,
        category="psf",
        description="Point Spread Function blur standard deviation (pixels)",
    ),
    "psf_size": ParameterDef(
        name="psf_size",
        bounds=(5, 21),
        dtype="int_odd",
        default=7,
        category="psf",
        description="PSF kernel size (odd integer)",
    ),
    # ── Noise ─────────────────────────────────────────────────────────────
    "peak_signal": ParameterDef(
        name="peak_signal",
        bounds=(500, 10000),
        dtype="float",
        default=8000.0,
        category="noise",
        description="Peak photon count for Poisson noise (higher = less noise)",
    ),
    "gaussian_sigma": ParameterDef(
        name="gaussian_sigma",
        bounds=(0.001, 0.05),
        dtype="float",
        default=0.01,
        category="noise",
        description="Gaussian readout noise standard deviation",
    ),
    # ── Brightness ────────────────────────────────────────────────────────
    "brightness_noise_strength": ParameterDef(
        name="brightness_noise_strength",
        bounds=(0.0, 0.5),
        dtype="float",
        default=0.1,
        category="brightness",
        description="Intra-cell brightness variation strength",
    ),
}


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def get_param_names() -> List[str]:
    """Return ordered list of all parameter names."""
    return list(PARAMETER_REGISTRY.keys())


def get_all_bounds() -> List[Tuple[float, float]]:
    """Return bounds for all parameters in registry order."""
    return [p.bounds for p in PARAMETER_REGISTRY.values()]


def get_all_defaults() -> Dict[str, Any]:
    """Return ``{name: default}`` for every registered parameter."""
    return {name: p.default for name, p in PARAMETER_REGISTRY.items()}


def get_bounds_for(names: List[str]) -> List[Tuple[float, float]]:
    """Return bounds for a specific subset of parameters (preserving order)."""
    return [PARAMETER_REGISTRY[n].bounds for n in names]


def get_defaults_for(names: List[str]) -> Dict[str, Any]:
    """Return defaults for a specific subset of parameters."""
    return {n: PARAMETER_REGISTRY[n].default for n in names}


def cast_param(name: str, value: float) -> Any:
    """Cast a parameter value to its declared dtype (int, int_odd, or float)."""
    pdef = PARAMETER_REGISTRY[name]
    if pdef.dtype == "int":
        return int(round(value))
    if pdef.dtype == "int_odd":
        v = int(round(value))
        if v % 2 == 0:
            lo, hi = pdef.bounds
            v = v + 1 if v + 1 <= hi else v - 1
        return v
    return float(value)


_DEFAULTS_CACHE: Dict[str, Any] | None = None


def _get_cached_defaults() -> Dict[str, Any]:
    """Return cached copy of registry defaults (allocated once)."""
    global _DEFAULTS_CACHE
    if _DEFAULTS_CACHE is None:
        _DEFAULTS_CACHE = get_all_defaults()
    return _DEFAULTS_CACHE


def build_full_params(
    active_names: List[str],
    active_values: List[float],
    fixed: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a complete parameter dict from active values + fixed defaults.

    Args:
        active_names: Names of the parameters being optimized.
        active_values: Corresponding values (same order as *active_names*).
        fixed: Explicit overrides for inactive parameters.  Missing keys
            fall back to registry defaults.

    Returns:
        Dict with a value for *every* registered parameter.
    """
    params = dict(_get_cached_defaults())
    if fixed:
        params.update(fixed)

    for name, val in zip(active_names, active_values):
        params[name] = cast_param(name, val)

    return params

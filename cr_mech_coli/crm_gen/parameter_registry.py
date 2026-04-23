"""
Central registry of all tunable parameters for synthetic microscope image generation.

Every parameter that can be optimized or screened is defined here with its
metadata (group, bounds, default, off_value, dtype).  This replaces the
hardcoded parameter lists.

To add a new parameter (e.g. for a new imaging modality):
    1.  Add a ``ParameterDef`` entry to ``PARAMETER_REGISTRY``.  Specify a
        ``group`` name — params sharing a group are perturbed together in
        grouped Morris screening.  Set ``off_value`` to the value that
        disables the effect (e.g. 0.0 for additive effects) if applicable.
    2.  Wire it into ``scene.py:apply_synthetic_effects()`` via the
        ``params`` dict pathway.
    3.  Re-run ``crm_gen screen`` — the screening will automatically
        evaluate whether the new parameter matters.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ParameterDef:
    """Definition of a single tunable parameter."""

    name: str
    group: str
    description: str
    bounds: Tuple[float, float]
    default: Any
    off_value: Optional[float] = None
    dtype: str = "float"  # "float", "int", or "int_odd"


# ---------------------------------------------------------------------------
# The registry — single source of truth for all optimizable parameters
# ---------------------------------------------------------------------------

PARAMETER_REGISTRY: Dict[str, ParameterDef] = {
    # ── Background ────────────────────────────────────────────────────────
    "bg_base_brightness": ParameterDef(
        name="bg_base_brightness",
        group="background",
        description="Base brightness level of the background",
        bounds=(0.0, 0.6),
        default=0.3,
    ),
    "bg_gradient_strength": ParameterDef(
        name="bg_gradient_strength",
        group="background",
        description="Illumination gradient intensity",
        bounds=(0.0, 0.04),
        default=0.001,
        off_value=0.0,
    ),
    "bg_noise_scale": ParameterDef(
        name="bg_noise_scale",
        group="background",
        description="Perlin-like background noise scale factor",
        bounds=(1, 25),
        default=5,
        dtype="int",
    ),
    "bg_blur_sigma": ParameterDef(
        name="bg_blur_sigma",
        group="background",
        description="Gaussian blur sigma for background optical effects",
        bounds=(0.1, 3.0),
        default=0.8,
    ),
    # ── Texture ───────────────────────────────────────────────────────────
    "texture_strength": ParameterDef(
        name="texture_strength",
        group="texture",
        description="Fine texture strength for background surface variations",
        bounds=(0.0, 0.1),
        default=0.02,
        off_value=0.0,
    ),
    "texture_scale": ParameterDef(
        name="texture_scale",
        group="texture",
        description="Texture smoothness (higher = smoother)",
        bounds=(0.5, 5.0),
        default=1.5,
    ),
    # ── Debris (dark spots) ───────────────────────────────────────────────
    "dark_spot_intensity": ParameterDef(
        name="dark_spot_intensity",
        group="debris",
        description="Intensity (darkness) of debris spots",
        bounds=(0.0, 0.3),
        default=0.15,
        off_value=0.0,
    ),
    "num_dark_spots_max": ParameterDef(
        name="num_dark_spots_max",
        group="debris",
        description="Maximum number of dark spots in background",
        bounds=(0, 10),
        default=0,
        off_value=0,
        dtype="int",
    ),
    # ── Halo ──────────────────────────────────────────────────────────────
    "bac_halo_intensity": ParameterDef(
        name="bac_halo_intensity",
        group="halo",
        description="Halo effect intensity around bacteria edges",
        bounds=(0.0, 0.6),
        default=0.0,
        off_value=0.0,
    ),
    "halo_inner_width": ParameterDef(
        name="halo_inner_width",
        group="halo",
        description="Width of inner bright halo in pixels",
        bounds=(0.5, 5.0),
        default=2.0,
    ),
    "halo_outer_width": ParameterDef(
        name="halo_outer_width",
        group="halo",
        description="Total halo extent in pixels",
        bounds=(5.0, 100.0),
        default=50.0,
    ),
    "halo_blur_sigma": ParameterDef(
        name="halo_blur_sigma",
        group="halo",
        description="Gaussian blur sigma for halo transition smoothing",
        bounds=(0.1, 3.0),
        default=0.5,
    ),
    # ── PSF ───────────────────────────────────────────────────────────────
    "psf_sigma": ParameterDef(
        name="psf_sigma",
        group="psf",
        description="Point Spread Function blur standard deviation (pixels)",
        bounds=(0.1, 3.0),
        default=1.0,
    ),
    "psf_size": ParameterDef(
        name="psf_size",
        group="psf",
        description="PSF kernel size (odd integer)",
        bounds=(5, 21),
        default=7,
        dtype="int_odd",
    ),
    # ── Poisson noise ─────────────────────────────────────────────────────
    "peak_signal": ParameterDef(
        name="peak_signal",
        group="poisson_noise",
        description="Peak photon count for Poisson noise (higher = less noise)",
        bounds=(500, 10000),
        default=8000.0,
    ),
    # ── Gaussian readout noise ────────────────────────────────────────────
    "gaussian_sigma": ParameterDef(
        name="gaussian_sigma",
        group="gaussian_noise",
        description="Gaussian readout noise standard deviation",
        bounds=(0.001, 0.05),
        default=0.01,
    ),
    # ── Brightness variation ──────────────────────────────────────────────
    "brightness_noise_strength": ParameterDef(
        name="brightness_noise_strength",
        group="brightness_variation",
        description="Intra-cell brightness variation strength",
        bounds=(0.0, 0.5),
        default=0.1,
        off_value=0.0,
    ),
    # ── Absorption (Beer-Lambert, brightfield) ────────────────────────────
    "absorption_coeff": ParameterDef(
        name="absorption_coeff",
        group="absorption",
        description="Beer-Lambert absorption coefficient for cell bodies",
        bounds=(0.0, 2.0),
        default=0.0,
        off_value=0.0,
    ),
    "cell_optical_thickness": ParameterDef(
        name="cell_optical_thickness",
        group="absorption",
        description="Maximum optical thickness at cell center (pixels)",
        bounds=(0.5, 10.0),
        default=3.0,
    ),
    # ── Defocus (depth-of-field blur) ─────────────────────────────────────
    "defocus_strength": ParameterDef(
        name="defocus_strength",
        group="defocus",
        description="Maximum additional blur sigma from defocus",
        bounds=(0.0, 3.0),
        default=0.0,
        off_value=0.0,
    ),
    "defocus_scale": ParameterDef(
        name="defocus_scale",
        group="defocus",
        description="Spatial scale of z-variation field for defocus",
        bounds=(1, 25),
        default=10,
        dtype="int",
    ),
    # ── Vignetting (radial illumination falloff) ──────────────────────────
    "vignette_strength": ParameterDef(
        name="vignette_strength",
        group="vignette",
        description="Radial intensity falloff strength from image center",
        bounds=(0.0, 0.5),
        default=0.0,
        off_value=0.0,
    ),
    # ── Edge diffraction fringe (brightfield) ─────────────────────────────
    "edge_fringe_intensity": ParameterDef(
        name="edge_fringe_intensity",
        group="edge_fringe",
        description="Strength of edge diffraction fringe at cell boundaries",
        bounds=(0.0, 0.1),
        default=0.0,
        off_value=0.0,
    ),
    "edge_fringe_width": ParameterDef(
        name="edge_fringe_width",
        group="edge_fringe",
        description="Width of edge diffraction fringe in pixels",
        bounds=(0.5, 4.0),
        default=1.5,
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


def get_groups_for(names: List[str]) -> List[str]:
    """Return the group name for each parameter (order-preserving)."""
    return [PARAMETER_REGISTRY[n].group for n in names]


def get_off_values_for(names: List[str]) -> Dict[str, Optional[float]]:
    """Return ``{name: off_value}`` for the given parameters.

    ``off_value`` is ``None`` for parameters whose effect cannot be meaningfully
    disabled (e.g. base brightness, PSF sigma — the image still has those).
    """
    return {n: PARAMETER_REGISTRY[n].off_value for n in names}


def get_params_by_group() -> Dict[str, List[str]]:
    """Return ``{group_name: [param_names]}`` for every group in the registry."""
    groups: Dict[str, List[str]] = {}
    for name, pdef in PARAMETER_REGISTRY.items():
        groups.setdefault(pdef.group, []).append(name)
    return groups


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

"""
End-to-end orchestration of synthetic training-data generation.

This module combines cr_mech_coli simulation with synthetic image generation:

    1. Samples simulation parameters
    2. Runs growth simulations & tracks cell lineages
    3. Renders raw images with segmentation masks
    4. Applies microscope-style post-processing
    5. Writes TIFF outputs with JSON metadata
    
"""

import os
os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')
os.environ.setdefault('VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN', '1')
# Force OSMesa software rendering (avoids EGL/GPU permission issues on clusters)
os.environ.setdefault('DISPLAY', '')
os.environ.setdefault('VTK_USE_OFFSCREEN_EGL', '0')

import json
import shutil
import multiprocessing as mp
from pathlib import Path
from functools import partial
from typing import Dict, List, Tuple, Any

import numpy as np
import tifffile as tiff
from tqdm import tqdm

import cr_mech_coli as crm
from .scene import apply_synthetic_effects


def sample_range(value: Any, rng: np.random.Generator) -> Any:
    """
    Sample a value from a range specification.

    Args:
        value: Either a scalar (returned as-is) or a [min, max] list (sampled uniformly).
        rng (np.random.Generator): NumPy random generator.

    Returns:
        Sampled or fixed value.
    """
    if isinstance(value, (list, tuple)) and len(value) == 2:
        min_val, max_val = value
        if min_val == max_val:
            return min_val
        return float(rng.uniform(min_val, max_val))
    return value


def sample_simulation_params(sim_config: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    """
    Sample concrete values from simulation parameter ranges.

    Args:
        sim_config (Dict[str, Any]): The [simulation] section from config with
            [min, max] ranges.
        rng (np.random.Generator): NumPy random generator.

    Returns:
        Dict[str, Any]: Dictionary with sampled concrete values.
    """
    sampled = {}
    for key, value in sim_config.items():
        sampled[key] = sample_range(value, rng)
    return sampled


# Aliases mapping registry-style param names to (section_label, section_key)
# for cases where the section TOML key differs from the registry name. Only
# the names listed here need translation; everything else matches its section
# key directly.
JITTER_ALIASES: Dict[str, Tuple[str, str]] = {
    "halo_inner_width": ("halo", "inner_width"),
    "halo_outer_width": ("halo", "outer_width"),
    "halo_blur_sigma": ("halo", "blur_sigma"),
    "bg_blur_sigma": ("background", "blur_sigma"),
    "brightness_noise_strength": ("brightness", "noise_strength"),
}


def apply_jitter(
    config_dicts: Dict[str, Dict[str, Any]],
    jitter_factors: Dict[str, float],
    rng: np.random.Generator,
) -> Dict[str, Dict[str, Any]]:
    """
    Apply per-parameter fractional jitter across multiple config sections.

    For each ``(name, factor)`` pair in ``jitter_factors`` with ``factor > 0``,
    resolve ``name`` to a ``(section, key)`` location and replace the value
    with one drawn uniformly from ``[base*(1 - factor), base*(1 + factor)]``.
    The result is clamped to non-negative when the base is non-negative.
    Non-numeric values (bool/str/list) are skipped silently. Unknown names
    (not in any section and not in ``JITTER_ALIASES``) are silently no-ops.

    Name resolution order:
      1. If ``name`` is in :data:`JITTER_ALIASES`, the alias's
         ``(section, key)`` is used.
      2. Otherwise the first section that contains ``name`` as a key wins.

    Args:
        config_dicts: Mapping from section label to its config dict, e.g.
            ``{"synthetic": ..., "background": ..., "halo": ..., "brightness": ...}``.
        jitter_factors: Mapping from param name to fractional variation
            (``0.05`` = +/-5%, ``0`` or missing = disabled).
        rng: NumPy random generator.

    Returns:
        Dict[str, Dict[str, Any]]: New dict-of-dicts (shallow-copied sections)
            with jittered values; inputs are not mutated.
    """
    out = {section: cfg.copy() for section, cfg in config_dicts.items()}
    if not jitter_factors:
        return out

    def _resolve(name: str):
        """Return (section_label, key) or None if the name can't be resolved."""
        if name in JITTER_ALIASES:
            section, key = JITTER_ALIASES[name]
            if section in out and key in out[section]:
                return section, key
            return None
        for section, cfg in out.items():
            if name in cfg:
                return section, name
        return None

    for param, factor in jitter_factors.items():
        try:
            factor = float(factor)
        except (TypeError, ValueError):
            continue
        if factor <= 0:
            continue
        location = _resolve(param)
        if location is None:
            continue
        section, key = location
        cfg = out[section]
        base = cfg[key]
        if isinstance(base, bool) or not isinstance(base, (int, float)):
            continue
        base = float(base)
        delta = abs(base) * factor
        new_val = float(rng.uniform(base - delta, base + delta))
        if base >= 0:
            new_val = max(0.0, new_val)
        cfg[key] = new_val

    return out


def load_parameter_sets(param_files: List[str]) -> List[Dict[str, Any]]:
    """
    Load parameter sets from JSON files.

    Args:
        param_files (List[str]): List of JSON file paths containing parameters.

    Returns:
        List[Dict[str, Any]]: List of parameter dictionaries.
    """
    param_sets = []
    for filepath in param_files:
        with open(filepath, 'r') as f:
            data = json.load(f)
            # Extract just the parameters dict
            if 'parameters' in data:
                param_sets.append(data['parameters'])
            else:
                param_sets.append(data)
    return param_sets


def compute_cell_ages(
    container: crm.CellContainer,
    iteration: int
) -> Dict[crm.CellIdentifier, int]:
    """
    Compute age for all cells at a specific iteration.

    Age is defined as: current_iteration - birth_iteration
    Daughter cells after division start at age 0.

    Args:
        container (crm.CellContainer): CellContainer from simulation.
        iteration (int): Target iteration to compute ages at.

    Returns:
        Dict[crm.CellIdentifier, int]: Dictionary mapping CellIdentifier to age
            (in iterations).
    """
    cells_at_iteration = container.get_cells_at_iteration(iteration)
    ages = {}

    for cell_id in cells_at_iteration.keys():
        cell_history, parent_id = container.get_cell_history(cell_id)
        # Birth iteration is the first iteration this cell appears
        birth_iteration = min(cell_history.keys())
        age = iteration - birth_iteration
        ages[cell_id] = age

    return ages


def run_simulation_image_gen(
    n_frames: int,
    image_size: Tuple[int, int],
    n_bacteria_range: Tuple[int, int],
    border_distance: float,
    max_bacteria_length: float,
    simulation_seed: int,
    n_vertices: int = 8,
    sim_params: Dict[str, Any] = None
) -> Tuple[crm.CellContainer, crm.Configuration]:
    """
    Run cr_mech_coli bacteria growth simulation.

    Args:
        n_frames (int): Number of frames (saved iterations) to generate.
        image_size (Tuple[int, int]): (width, height) of the simulation domain in pixels.
        n_bacteria_range (Tuple[int, int]): (min, max) range for initial bacteria count.
        border_distance (float): Minimum distance from border for initial positions.
        max_bacteria_length (float): Maximum bacteria length before division.
        simulation_seed (int): Random seed for simulation.
        n_vertices (int): Number of vertices per bacterium.
        sim_params (Dict[str, Any]): Optional dict of sampled simulation parameters
            to override defaults.

    Returns:
        Tuple[crm.CellContainer, crm.Configuration]: CellContainer with simulation
            results and Configuration used.
    """
    if sim_params is None:
        sim_params = {}

    # Set up configuration
    config = crm.Configuration()
    config.domain_size = (float(image_size[0]), float(image_size[1]))
    config.n_saves = n_frames
    config.rng_seed = simulation_seed
    config.storage_options = [crm.StorageOption.Memory]

    # Apply simulation timing params from config
    if 't_max' in sim_params:
        config.t_max = float(sim_params['t_max'])
    if 'dt' in sim_params:
        config.dt = float(sim_params['dt'])
    if 'gel_pressure' in sim_params:
        config.gel_pressure = float(sim_params['gel_pressure'])
    if 'surface_friction' in sim_params:
        config.surface_friction = float(sim_params['surface_friction'])
    if 'domain_height' in sim_params:
        config.domain_height = float(sim_params['domain_height'])

    # Build AgentSettings kwargs - some params like growth_rate_setter must be set at construction
    agent_kwargs = {}

    # Growth rate and its distribution at division
    growth_rate = sim_params.get('growth_rate', 0.01)  # cr_mech_coli default
    agent_kwargs['growth_rate'] = float(growth_rate)

    # growth_rate_setter must be set at construction time (immutable after)
    growth_rate_std = sim_params.get('growth_rate_std', 0.0)
    if growth_rate_std > 0:
        agent_kwargs['growth_rate_setter'] = {
            'mean': float(growth_rate),
            'std': float(growth_rate_std)
        }

    # Spring length threshold (cell length at division)
    spring_length_threshold = sim_params.get('spring_length_threshold', max_bacteria_length)
    agent_kwargs['spring_length_threshold'] = float(spring_length_threshold)

    # Create agent settings with the kwargs that must be set at construction
    agent_settings = crm.AgentSettings(**agent_kwargs)

    # Adjust spring length for vertex count (after construction)
    agent_settings.mechanics.spring_length *= 8 / n_vertices

    # Mechanics params (can be set after construction)
    if 'diffusion_constant' in sim_params:
        agent_settings.mechanics.diffusion_constant = float(sim_params['diffusion_constant'])
    if 'spring_tension' in sim_params:
        agent_settings.mechanics.spring_tension = float(sim_params['spring_tension'])
    if 'rigidity' in sim_params:
        agent_settings.mechanics.rigidity = float(sim_params['rigidity'])
    if 'damping' in sim_params:
        agent_settings.mechanics.damping = float(sim_params['damping'])

    # Interaction params (Morse potential)
    # Note: Only 'strength' is writable via Python API.
    if 'interaction_strength' in sim_params:
        agent_settings.interaction.strength = float(sim_params['interaction_strength'])
    if 'radius' in sim_params:
        agent_settings.interaction.radius = float(sim_params['radius'])

    # Determine number of bacteria
    rng = np.random.default_rng(simulation_seed)
    if n_bacteria_range[0] == n_bacteria_range[1]:
        n_bacteria = n_bacteria_range[0]
    else:
        n_bacteria = rng.integers(n_bacteria_range[0], n_bacteria_range[1] + 1)

    # Generate initial agents with border distance
    agents = crm.generate_agents(
        n_bacteria,
        agent_settings,
        config,
        rng_seed=simulation_seed,
        dx=(border_distance, border_distance),
        n_vertices=n_vertices
    )

    print(f"Running simulation with {n_bacteria} bacteria...")
    container = crm.run_simulation_with_agents(config, agents)

    return container, config


def render_and_save_frame(
    container: crm.CellContainer,
    iteration: int,
    domain_size: Tuple[float, float],
    output_dir: Path,
    render_settings: crm.RenderSettings,
    file_prefix: str = "",
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Render and save a single frame (image and mask).

    Args:
        container (crm.CellContainer): Simulation results container.
        iteration (int): Iteration number to render.
        domain_size (Tuple[float, float]): Simulation domain size.
        output_dir (Path): Directory to save output files.
        render_settings (crm.RenderSettings): Rendering configuration.
        file_prefix (str): Prefix prepended to every output filename (e.g.
            ``"sim_0003_"``). Empty string disables prefixing.

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict]: (image, mask, cell_colors_dict).
    """
    cells = container.get_cells_at_iteration(iteration)
    colors = container.cell_to_color

    # Render image
    image = crm.render_image(
        cells,
        domain_size=domain_size,
        render_settings=render_settings
    )

    # Render mask
    mask = crm.render_mask(
        cells,
        colors=colors,
        domain_size=domain_size,
        render_settings=render_settings
    )

    # Save files
    frame_name = f"{iteration:09d}"
    image_path = output_dir / f"{file_prefix}{frame_name}.tif"
    mask_path = output_dir / f"{file_prefix}{frame_name}_masks.tif"

    tiff.imwrite(image_path, image, compression='zlib')
    tiff.imwrite(mask_path, mask, compression='zlib')

    return image, mask, colors


def process_frame_for_synthetic(args):
    """
    Worker function for parallel synthetic image generation.

    Args:
        args: Tuple of (frame_idx, iteration, generated_dir, synthetic_dir,
            cell_ages, cell_colors_serializable, synthetic_config,
            background_config, halo_config, brightness_config,
            jitter_factors, delete_after_processing, bg_seed, file_prefix).
            When ``jitter_factors`` is empty, the four config dicts are used
            as-is (already jittered at sim level, or no jitter requested).

    Returns:
        Tuple[int, int, Dict]: (frame_idx, iteration, params) with the parameters
            used for this frame.
    """
    (
        frame_idx,
        iteration,
        generated_dir,
        synthetic_dir,
        cell_ages,
        cell_colors_serializable,
        synthetic_config,
        background_config,
        halo_config,
        brightness_config,
        jitter_factors,
        delete_after_processing,
        bg_seed,
        file_prefix,
    ) = args

    # Reconstruct cell_colors dict (keys were serialized)
    cell_colors = {}
    for key_str, color in cell_colors_serializable.items():
        cell_colors[key_str] = tuple(color)

    # Load generated image and mask
    frame_name = f"{iteration:09d}"
    image_path = generated_dir / f"{file_prefix}{frame_name}.tif"
    mask_path = generated_dir / f"{file_prefix}{frame_name}_masks.tif"
    image = tiff.imread(image_path)
    mask = tiff.imread(mask_path)

    # Generate random seed for this specific frame
    seed = np.random.default_rng().integers(0, 2**31)
    rng = np.random.default_rng(seed)

    # Per-frame jitter (no-op when jitter_factors is empty, e.g. when the
    # sim-level branch already froze the values for consistent backgrounds).
    jittered = apply_jitter(
        {
            "synthetic": synthetic_config,
            "background": background_config,
            "halo": halo_config,
            "brightness": brightness_config,
        },
        jitter_factors,
        rng,
    )
    synthetic_params = jittered["synthetic"]
    background_params = jittered["background"]
    halo_params = jittered["halo"]
    brightness_params = jittered["brightness"]

    # Extract parameters with defaults
    bg_base_brightness = synthetic_params.get('bg_base_brightness', 0.56)
    bg_gradient_strength = synthetic_params.get('bg_gradient_strength', 0.027)
    bac_halo_intensity = synthetic_params.get('bac_halo_intensity', 0.30)
    bg_noise_scale = int(synthetic_params.get('bg_noise_scale', 20))  # Must be int
    psf_sigma = synthetic_params.get('psf_sigma', 1.0)
    peak_signal = synthetic_params.get('peak_signal', 1000.0)
    gaussian_sigma = synthetic_params.get('gaussian_sigma', 0.01)

    # Effect toggles
    apply_psf = synthetic_params.get('apply_psf', True)
    apply_poisson = synthetic_params.get('apply_poisson', True)
    apply_gaussian = synthetic_params.get('apply_gaussian', True)

    # Background params
    bg_dark_spot_size_range = tuple(background_params.get('dark_spot_size_range', [2, 5]))
    bg_num_dark_spots_range = tuple(background_params.get('num_dark_spots_range', [0, 5]))
    # Registry-style upper bound; when present, it overrides num_dark_spots_range[1].
    if 'num_dark_spots_max' in background_params:
        bg_num_dark_spots_range = (0, int(background_params['num_dark_spots_max']))
    bg_num_light_spots_range = tuple(background_params.get('num_light_spots_range', [0, 0]))
    bg_texture_strength = background_params.get('texture_strength', 0.02)
    bg_texture_scale = background_params.get('texture_scale', 1.5)
    bg_blur_sigma = background_params.get('blur_sigma', 0.8)
    dark_spot_intensity = background_params.get('dark_spot_intensity', 0.0)

    # Halo params
    halo_inner_width = halo_params.get('inner_width', 2.0)
    halo_outer_width = halo_params.get('outer_width', 50.0)
    halo_blur_sigma = halo_params.get('blur_sigma', 0.5)
    halo_fade_type = halo_params.get('fade_type', 'exponential')

    # Brightness params
    brightness_mode = brightness_params.get('mode', 'age')
    brightness_range = tuple(brightness_params.get('brightness_range', [0.8, 0.3]))
    max_age = brightness_params.get('max_age')
    brightness_noise_strength = brightness_params.get('noise_strength', 0.0)

    # Extended optical params (registry-mapped, zero-value disable except for
    # parent-gated children like cell_optical_thickness/defocus_scale).
    psf_size = int(synthetic_params.get('psf_size', 7))
    absorption_coeff = synthetic_params.get('absorption_coeff', 0.0)
    cell_optical_thickness = synthetic_params.get('cell_optical_thickness', 3.0)
    defocus_strength = synthetic_params.get('defocus_strength', 0.0)
    defocus_scale = int(synthetic_params.get('defocus_scale', 10))
    vignette_strength = synthetic_params.get('vignette_strength', 0.0)
    edge_fringe_intensity = synthetic_params.get('edge_fringe_intensity', 0.0)
    edge_fringe_width = synthetic_params.get('edge_fringe_width', 1.5)

    # Create synthetic image using shared function
    synthetic_image = apply_synthetic_effects(
        raw_image=image,
        mask=mask,
        bg_base_brightness=bg_base_brightness,
        bg_gradient_strength=bg_gradient_strength,
        bac_halo_intensity=bac_halo_intensity,
        bg_noise_scale=bg_noise_scale,
        psf_sigma=psf_sigma,
        peak_signal=peak_signal,
        gaussian_sigma=gaussian_sigma,
        seed=seed,
        brightness_mode=brightness_mode,
        cell_ages=cell_ages,
        cell_colors_map=cell_colors,
        brightness_range=brightness_range,
        max_age=max_age,
        num_dark_spots_range=bg_num_dark_spots_range,
        # Effect toggles
        apply_psf=apply_psf,
        apply_poisson=apply_poisson,
        apply_gaussian=apply_gaussian,
        # Background params
        dark_spot_size_range=bg_dark_spot_size_range,
        num_light_spots_range=bg_num_light_spots_range,
        texture_strength=bg_texture_strength,
        texture_scale=bg_texture_scale,
        bg_blur_sigma=bg_blur_sigma,
        # Halo params
        halo_inner_width=halo_inner_width,
        halo_outer_width=halo_outer_width,
        halo_blur_sigma=halo_blur_sigma,
        halo_fade_type=halo_fade_type,
        # Brightness noise
        brightness_noise_strength=brightness_noise_strength,
        # Consistent background across frames
        bg_seed=bg_seed,
        # Extended registry params
        dark_spot_intensity=dark_spot_intensity,
        psf_size=psf_size,
        absorption_coeff=absorption_coeff,
        cell_optical_thickness=cell_optical_thickness,
        defocus_strength=defocus_strength,
        defocus_scale=defocus_scale,
        vignette_strength=vignette_strength,
        edge_fringe_intensity=edge_fringe_intensity,
        edge_fringe_width=edge_fringe_width,
    )

    # Save synthetic image and copy mask
    output_prefix = f"{file_prefix}syn_{frame_name}"
    tiff.imwrite(
        synthetic_dir / f"{output_prefix}.tif",
        synthetic_image,
        compression='zlib'
    )
    tiff.imwrite(
        synthetic_dir / f"{output_prefix}_masks.tif",
        mask,
        compression='zlib'
    )

    # Delete generated files if requested
    if delete_after_processing:
        if image_path.exists():
            image_path.unlink()
        if mask_path.exists():
            mask_path.unlink()

    return frame_idx, iteration, synthetic_params


def run_pipeline(
    output_dir: str = "./outputs",
    n_frames: int = 10,
    image_size: Tuple[int, int] = (512, 512),
    n_bacteria_range: Tuple[int, int] = (1, 10),
    border_distance: float = 5.0,
    max_bacteria_length: float = 6.0,
    simulation_seed: int = None,
    n_vertices: int = 8,
    parameter_sets: List[str] = None,
    brightness_range: Tuple[float, float] = (0.8, 0.3),
    num_dark_spots_range: Tuple[int, int] = (0, 5),
    skip_synthetic: bool = False,
    delete_generated: bool = False,
    n_workers: int = None,
    # TOML config parameters
    sim_param_ranges: Dict[str, Any] = None,
    rendering_config: Dict[str, Any] = None,
    synthetic_config: Dict[str, Any] = None,
    background_config: Dict[str, Any] = None,
    halo_config: Dict[str, Any] = None,
    brightness_config: Dict[str, Any] = None,
    jitter_factors: Dict[str, float] = None,
    n_simulations: int = 1
):
    """
    Run the complete synthetic image generation pipeline.

    Args:
        output_dir (str): Directory to save outputs.
        n_frames (int): Number of frames to generate.
        image_size (Tuple[int, int]): (width, height) of images in pixels.
        n_bacteria_range (Tuple[int, int]): (min, max) range for initial bacteria count.
        border_distance (float): Minimum distance from border for bacteria.
        max_bacteria_length (float): Max bacteria length before division.
        simulation_seed (int): Random seed. If None, generates a random seed.
        n_vertices (int): Number of vertices per bacterium.
        parameter_sets (List[str]): List of JSON file paths with parameters (legacy).
        brightness_range (Tuple[float, float]): (young, old) brightness for age-based
            mode (legacy).
        num_dark_spots_range (Tuple[int, int]): (min, max) range for dark spots (legacy).
        skip_synthetic (bool): Only run simulation, skip synthetic generation.
        delete_generated (bool): Delete raw generated images after processing.
        n_workers (int): Number of parallel workers. If None, uses all CPUs.
        sim_param_ranges (Dict[str, Any]): Simulation parameter ranges from TOML
            [simulation] section.
        rendering_config (Dict[str, Any]): Rendering settings from TOML [rendering]
            section.
        synthetic_config (Dict[str, Any]): Synthetic params from TOML [synthetic] section.
        background_config (Dict[str, Any]): Background params from TOML [background]
            section.
        halo_config (Dict[str, Any]): Halo params from TOML [halo] section.
        brightness_config (Dict[str, Any]): Brightness params from TOML [brightness]
            section.
        n_simulations (int): Number of simulations with different randomized parameters.
    """
    # Initialize default configs if not provided
    if sim_param_ranges is None:
        sim_param_ranges = {}
    if rendering_config is None:
        rendering_config = {}
    if synthetic_config is None:
        synthetic_config = {}
    if background_config is None:
        background_config = {'num_dark_spots_range': list(num_dark_spots_range)}
    if halo_config is None:
        halo_config = {}
    if brightness_config is None:
        brightness_config = {'brightness_range': list(brightness_range)}
    if jitter_factors is None:
        jitter_factors = {}

    # Setup base output directory
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Initialize base seed
    if simulation_seed is None:
        simulation_seed = np.random.default_rng().integers(0, 2**31)

    if n_workers is None or n_workers == 0:
        n_workers = mp.cpu_count()

    # Count the number of params that will actually be jittered (factor > 0)
    active_jitter = sum(
        1 for v in jitter_factors.values()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and float(v) > 0
    )

    # Load legacy parameter sets if provided (for backwards compatibility)
    param_sets = []
    if parameter_sets and not skip_synthetic:
        param_sets = load_parameter_sets(parameter_sets)
        print(f"Loaded {len(param_sets)} parameter sets (legacy mode)")

    # Create base RNG for deriving per-simulation seeds
    base_rng = np.random.default_rng(simulation_seed)

    print(f"\n{'='*60}")
    print(f"Pipeline Configuration")
    print(f"{'='*60}")
    print(f"  Simulations: {n_simulations}")
    print(f"  Frames per simulation: {n_frames}")
    print(f"  Image size: {image_size}")
    print(f"  Base seed: {simulation_seed}")
    print(f"  Workers: {n_workers}")
    print(f"  Jitter: {active_jitter} param(s) active")
    print(f"{'='*60}")

    # Main loop over simulations
    for sim_idx in range(n_simulations):
        # Derive per-simulation seed
        sim_seed = int(base_rng.integers(0, 2**31))
        sim_rng = np.random.default_rng(sim_seed)

        # Sample simulation parameters from ranges
        sampled_sim_params = sample_simulation_params(sim_param_ranges, sim_rng)

        # Override specific params from sampled values
        n_bacteria_actual = n_bacteria_range
        if 'n_bacteria' in sampled_sim_params:
            val = sampled_sim_params['n_bacteria']
            if isinstance(val, (list, tuple)):
                n_bacteria_actual = (int(val[0]), int(val[1]))
            else:
                n_bacteria_actual = (int(val), int(val))

        border_dist_actual = sampled_sim_params.get('border_distance', border_distance)
        n_verts_actual = int(sampled_sim_params.get('n_vertices', n_vertices))
        max_len_actual = sampled_sim_params.get('spring_length_threshold', max_bacteria_length)

        # All simulations share one generated/ and one synthetic/ directory; per-sim
        # files are distinguished by the file_prefix instead of a per-sim subdir.
        file_prefix = f"sim_{sim_idx:04d}_" if n_simulations > 1 else ""
        generated_dir = output_base / "generated"
        synthetic_dir = output_base / "synthetic"

        generated_dir.mkdir(parents=True, exist_ok=True)
        if not skip_synthetic:
            synthetic_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        if n_simulations > 1:
            print(f"Simulation {sim_idx + 1}/{n_simulations}")
        print(f"{'='*60}")

        # Step 1: Run simulation
        print(f"\n=== Step 1: Running simulation ===")
        print(f"  Frames: {n_frames}")
        print(f"  Image size: {image_size}")
        print(f"  Bacteria range: {n_bacteria_actual}")
        print(f"  Border distance: {border_dist_actual}")
        print(f"  Seed: {sim_seed}")
        if sampled_sim_params:
            print(f"  Sampled parameters:")
            for key, val in sampled_sim_params.items():
                if key not in ('n_bacteria', 'border_distance', 'n_vertices'):
                    print(f"    {key}: {val}")

        container, config = run_simulation_image_gen(
            n_frames=n_frames,
            image_size=image_size,
            n_bacteria_range=n_bacteria_actual,
            border_distance=border_dist_actual,
            max_bacteria_length=max_len_actual,
            simulation_seed=sim_seed,
            n_vertices=n_verts_actual,
            sim_params=sampled_sim_params
        )

        iterations = container.get_all_iterations()
        print(f"  Generated {len(iterations)} iterations")

        # Step 2: Compute cell ages for all iterations
        print(f"\n=== Step 2: Computing cell ages ===")
        all_ages = {}
        max_age_overall = 0

        for iteration in iterations:
            ages = compute_cell_ages(container, iteration)
            all_ages[iteration] = ages
            if ages:
                max_age_overall = max(max_age_overall, max(ages.values()))

        print(f"  Max age across all frames: {max_age_overall}")

        # Step 3: Render and save generated images
        print(f"\n=== Step 3: Rendering generated images ===")

        render_settings = crm.RenderSettings()
        render_settings.pixel_per_micron = rendering_config.get('pixel_per_micron', 1.0)
        render_settings.kernel_size = rendering_config.get('kernel_size', 2)
        render_settings.noise = rendering_config.get('noise', 0)
        render_settings.bg_brightness = rendering_config.get('bg_brightness', 0)
        render_settings.cell_brightness = rendering_config.get('cell_brightness', 0)
        render_settings.ambient = rendering_config.get('ambient', 0.3)
        render_settings.diffuse = rendering_config.get('diffuse', 0.7)
        render_settings.specular = rendering_config.get('specular', 0.0)
        render_settings.specular_power = rendering_config.get('specular_power', 0.0)
        render_settings.metallic = rendering_config.get('metallic', 0.0)
        render_settings.pbr = rendering_config.get('pbr', False)
        render_settings.ssao_radius = rendering_config.get('ssao_radius', 0.0)

        cell_colors_per_iteration = {}

        for iteration in tqdm(iterations, desc="Rendering frames"):
            _, _, colors = render_and_save_frame(
                container=container,
                iteration=iteration,
                domain_size=config.domain_size,
                output_dir=generated_dir,
                render_settings=render_settings,
                file_prefix=file_prefix,
            )
            cell_colors_per_iteration[iteration] = colors

        print(f"  Saved {len(iterations)} frames to {generated_dir}")

        # Update brightness_config with max_age
        brightness_config_with_max_age = brightness_config.copy()
        brightness_config_with_max_age['max_age'] = max_age_overall

        # Derive a consistent background seed for this simulation
        # When background.consistent is true (default), all frames share the
        # same background; otherwise each frame gets its own random background.
        consistent_bg = background_config.get('consistent', True)
        bg_seed = int(sim_rng.integers(0, 2**31)) if consistent_bg else None

        # Jitter cadence:
        #   consistent_bg=True  -> draw jitter ONCE for this sim using sim_rng;
        #                          workers receive frozen configs + empty factors.
        #   consistent_bg=False -> workers re-jitter every frame; pass original
        #                          configs + full factors.
        if consistent_bg and active_jitter > 0:
            frozen = apply_jitter(
                {
                    "synthetic": synthetic_config,
                    "background": background_config,
                    "halo": halo_config,
                    "brightness": brightness_config_with_max_age,
                },
                jitter_factors,
                sim_rng,
            )
            worker_synthetic = frozen["synthetic"]
            worker_background = frozen["background"]
            worker_halo = frozen["halo"]
            worker_brightness = frozen["brightness"]
            worker_jitter_factors = {}
        else:
            worker_synthetic = synthetic_config
            worker_background = background_config
            worker_halo = halo_config
            worker_brightness = brightness_config_with_max_age
            worker_jitter_factors = jitter_factors

        # Step 4: Save metadata
        metadata = {
            "simulation_seed": sim_seed,
            "base_seed": simulation_seed,
            "simulation_index": sim_idx,
            "n_frames": n_frames,
            "image_size": list(image_size),
            "n_bacteria_range": list(n_bacteria_actual),
            "border_distance": border_dist_actual,
            "n_vertices": n_verts_actual,
            "iterations": iterations,
            "max_age": max_age_overall,
            "sampled_sim_params": sampled_sim_params,
            "synthetic_config": worker_synthetic,
            "background_config": worker_background,
            "halo_config": worker_halo,
            "brightness_config": worker_brightness,
            "rendering_config": rendering_config,
            "consistent_background": consistent_bg,
            "bg_seed": bg_seed,
            "jitter_factors": jitter_factors,
            "jitter_frozen_per_sim": bool(consistent_bg and active_jitter > 0),
            "ages_per_iteration": {
                str(it): {str(k): v for k, v in ages.items()}
                for it, ages in all_ages.items()
            }
        }

        with open(output_base / f"{file_prefix}metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        if skip_synthetic:
            print(f"\n=== Skipping synthetic image generation ===")
            print(f"Simulation complete. Generated images saved to: {generated_dir}")
            continue

        # Check if we have synthetic config or legacy param_sets
        if not synthetic_config and not param_sets:
            print(f"\n=== No synthetic parameters provided, skipping synthetic generation ===")
            continue

        # Step 5: Generate synthetic images
        print(f"\n=== Step 4: Generating synthetic images ===")
        print(f"  Total images to generate: {len(iterations)}")
        print(f"  Workers: {n_workers}")
        if active_jitter > 0:
            cadence = "frozen per-sim" if consistent_bg else "per-frame"
            print(f"  Jitter: {active_jitter} param(s), {cadence}")

        # Prepare arguments for parallel processing
        work_items = []
        for frame_idx, iteration in enumerate(iterations):
            # Serialize cell colors for multiprocessing
            colors = cell_colors_per_iteration[iteration]
            colors_serializable = {str(k): list(v) for k, v in colors.items()}

            # Serialize ages
            ages_serializable = {str(k): v for k, v in all_ages[iteration].items()}

            work_items.append((
                frame_idx,
                iteration,
                generated_dir,
                synthetic_dir,
                ages_serializable,
                colors_serializable,
                worker_synthetic,
                worker_background,
                worker_halo,
                worker_brightness,
                worker_jitter_factors,
                delete_generated,
                bg_seed,
                file_prefix,
            ))

        # Process in parallel
        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                results = list(tqdm(
                    pool.imap_unordered(process_frame_for_synthetic, work_items),
                    total=len(work_items),
                    desc="Generating synthetic images"
                ))
        else:
            results = []
            for item in tqdm(work_items, desc="Generating synthetic images"):
                results.append(process_frame_for_synthetic(item))

        print(f"  Saved synthetic images to {synthetic_dir}")

    # Final cleanup: remove the shared generated/ directory once all sims have
    # finished (workers delete their own files per-frame when delete_generated
    # is set, so this just removes the now-empty dir).
    if delete_generated and not skip_synthetic and generated_dir.exists():
        shutil.rmtree(generated_dir, ignore_errors=True)
        print(f"  Cleaned up {generated_dir}")

    print(f"\n{'='*60}")
    print(f"Pipeline complete")
    print(f"{'='*60}")
    print(f"Output directory: {output_base}")

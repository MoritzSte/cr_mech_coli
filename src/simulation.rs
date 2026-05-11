use backend::chili::SimulationError;
use cellular_raza::prelude::*;
use numpy::ToPyArray;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::cell_container::CellContainer;

use crate::agent::*;
use crate::config::*;

prepare_types!(
    aspects: [Mechanics, Interaction, Cycle, DomainForce, NeighborSensing],
);

/// Creates positions for multiple :class:`RodAgent` which can be used for simulation purposes.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    n_agents,
    agent_settings,
    config,
    rng_seed = 0,
    dx = [0.0, 0.0],
    randomize_positions = 0.0,
    n_vertices = 8,
))]
pub fn generate_positions<'py>(
    py: Python<'py>,
    n_agents: usize,
    agent_settings: &crate::config::AgentSettings,
    config: &Configuration,
    rng_seed: u64,
    dx: [f32; 2],
    randomize_positions: f32,
    n_vertices: usize,
) -> PyResult<Vec<Bound<'py, numpy::PyArray2<f32>>>> {
    let mechanics: RodMechanicsSettings = agent_settings.mechanics.extract(py)?;
    Ok(_generate_positions(
        n_agents,
        &mechanics,
        config,
        rng_seed,
        dx,
        randomize_positions,
        n_vertices,
    )
    .into_iter()
    .map(|x| x.to_pyarray(py))
    .collect())
}

/// Uses the :func:`generate_positions` function to generate positions and then sets parameters
/// of the generated agents from the supplied :class:`AgentSettings`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    n_agents,
    agent_settings,
    config,
    rng_seed = 0,
    dx = [0.0, 0.0],
    randomize_positions = 0.0,
    n_vertices = 8,
))]
pub fn generate_agents(
    py: Python,
    n_agents: usize,
    agent_settings: &AgentSettings,
    config: &Configuration,
    rng_seed: u64,
    dx: [f32; 2],
    randomize_positions: f32,
    n_vertices: usize,
) -> Vec<crate::RodAgent> {
    use core::ops::Deref;
    let positions = _generate_positions(
        n_agents,
        agent_settings.mechanics.borrow(py).deref(),
        config,
        rng_seed,
        dx,
        randomize_positions,
        n_vertices,
    );
    positions
        .into_iter()
        .map(|pos| {
            let n_v = pos.nrows();
            crate::RodAgent {
            mechanics: RodMechanics {
                vel: nalgebra::MatrixXx3::zeros(n_v),
                pos,
                ..agent_settings.mechanics.borrow(py).clone().into()
            },
            interaction: RodInteraction(agent_settings.interaction.borrow(py).clone()),
            growth_rate: agent_settings.growth_rate,
            growth_rate_setter: agent_settings.growth_rate_setter.borrow(py).clone(),
            spring_length_threshold: agent_settings.spring_length_threshold,
            spring_length_threshold_setter: agent_settings
                .spring_length_threshold_setter
                .borrow(py)
                .clone(),
            neighbor_reduction: agent_settings.neighbor_reduction,
        }
        })
        .collect()
}

/// Backend functionality to use within rust-specific code for [generate_positions]
pub fn _generate_positions(
    n_agents: usize,
    mechanics: &RodMechanicsSettings,
    config: &Configuration,
    rng_seed: u64,
    dx: [f32; 2],
    randomize_positions: f32,
    n_vertices: usize,
) -> Vec<numpy::nalgebra::MatrixXx3<f32>> {
    // numpy::nalgebra::DMatrix<f32>
    use rand::seq::IteratorRandom;
    use rand::Rng;
    use rand_chacha::rand_core::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(rng_seed);
    let spring_length = mechanics.spring_length;
    let s = randomize_positions.clamp(0.0, 1.0);

    // Split the domain into chunks
    let n_chunk_sides = (n_agents as f32).sqrt().ceil() as usize;
    let dchunk1 = (config.domain_size[0] - 2.0 * dx[0]) / n_chunk_sides as f32;
    let dchunk2 = (config.domain_size[1] - 2.0 * dx[1]) / n_chunk_sides as f32;
    let all_indices = itertools::iproduct!(0..n_chunk_sides, 0..n_chunk_sides);
    let picked_indices = all_indices.choose_multiple(&mut rng, n_agents);
    let drod_length_half = (n_vertices as f32) * spring_length / 2.0;

    picked_indices
        .into_iter()
        .map(|index| {
            let xlow = dx[0] + index.0 as f32 * dchunk1;
            let ylow = dx[1] + index.1 as f32 * dchunk2;
            let middle = numpy::array![
                rng.random_range(xlow + drod_length_half..xlow + dchunk1 - drod_length_half),
                rng.random_range(ylow + drod_length_half..ylow + dchunk2 - drod_length_half),
                rng.random_range(0.4 * config.domain_height..0.6 * config.domain_height),
            ];
            let angle: f32 = rng.random_range(0.0..2.0 * std::f32::consts::PI);
            let p1 = middle - drod_length_half * numpy::array![angle.cos(), angle.sin(), 0.0];
            fn s_gen(x: f32, rng: &mut rand_chacha::ChaCha8Rng) -> f32 {
                if x == 0.0 {
                    1.0
                } else {
                    rng.random_range(1.0 - x..1.0 + x)
                }
            }
            numpy::nalgebra::MatrixXx3::<f32>::from_fn(n_vertices, |r, c| {
                p1[c]
                    + r as f32
                        * spring_length
                        * s_gen(s, &mut rng)
                        * if c == 0 {
                            (angle * s_gen(s, &mut rng)).cos()
                        } else if c == 1 {
                            (angle * s_gen(s, &mut rng)).sin()
                        } else {
                            0.0
                        }
            })
        })
        .collect()
}

#[test]
fn backwards_compat_generate_positions() -> PyResult<()> {
    let mechanics = RodMechanicsSettings::default();
    let config = Configuration::default();
    let generated_pos = _generate_positions(4, &mechanics, &config, 1, [0.0; 2], 0.1, 8);
    let old_pos = vec![
        numpy::nalgebra::dmatrix![
            15.782119,  16.658249,  1.4922986;
            18.387316,  18.2603,    1.4922986;
            21.448421,  19.527,     1.4922986;
            24.20771,   21.220747,  1.4922986;
            26.298336,  22.355158,  1.4922986;
            29.717592,  23.55539,   1.4922986;
            29.781338,  25.25969,   1.4922986;
            32.75929,   28.03757,   1.4922986;
        ],
        numpy::nalgebra::dmatrix![
             4.2639103, 71.299194,  1.4667264;
             7.1084356, 71.53958,   1.4667264;
            10.703307,  71.7947,    1.4667264;
            13.803044,  72.10863,   1.4667264;
            15.5561495, 72.22104,   1.4667264;
            18.084274,  72.68125,   1.4667264;
            20.477375,  72.87153,   1.4667264;
            23.981491,  73.02278,   1.4667264;
        ],
        numpy::nalgebra::dmatrix![
            68.69818,   30.033642,  1.384213;
            71.653015,  29.911337,  1.384213;
            72.82722,   28.599209,  1.384213;
            74.60274,   26.107859,  1.384213;
            80.05429,   26.160421,  1.384213;
            84.007164,  24.251816,  1.384213;
            83.433105,  23.3395,    1.384213;
            90.09657,   18.543726,  1.384213;
        ],
        numpy::nalgebra::dmatrix![
            89.117294,  63.976006,  1.4164526;
            87.471756,  66.64026,   1.4164526;
            84.89454,   67.19988,   1.4164526;
            84.3147,    70.79428,   1.4164526;
            80.57336,   70.544975,  1.4164526;
            79.02444,   75.27408,   1.4164526;
            78.291405,  79.85086,   1.4164526;
            74.64508,   82.48188,   1.4164526;
        ],
    ];
    for (p, q) in generated_pos.into_iter().zip(old_pos.into_iter()) {
        assert_eq!(p, q);
    }
    Ok(())
}

pub(crate) fn new_storage_builder(config: &Configuration) -> StorageBuilder {
    let builder = StorageBuilder::new()
        .priority(config.storage_options.clone())
        .location(&config.storage_location);
    if let Some(suffix) = &config.storage_suffix {
        builder.suffix(suffix)
    } else {
        builder
    }
}

/// Executes a simulation given a :class:`Configuration` and a list of :class:`RodAgent`.
#[pyfunction]
pub fn run_simulation_with_agents(
    config: &Configuration,
    agents: Vec<RodAgent>,
) -> Result<CellContainer, cellular_raza::prelude::SimulationError> {
    // TODO after initializing this state, we need to check that it is actually valid
    let t0 = config.t0;
    let dt = config.dt;
    let t_max = config.t_max;
    let mut save_steps = vec![t0];
    if config.n_saves > 0 {
        let dtsave = (t_max - t0) / (config.n_saves + 1) as f32;
        save_steps.extend((1..config.n_saves + 1).map(|n| t0 + n as f32 * dtsave));
    }
    save_steps.push(t_max);
    let time = FixedStepsize::from_partial_save_points(t0, dt, save_steps)
        .map_err(SimulationError::from)?;
    let storage = new_storage_builder(config);
    let settings = Settings {
        n_threads: config.n_threads,
        time,
        storage,
        progressbar: config.progressbar.clone(),
    };

    let domain_size = [
        config.domain_size[0],
        config.domain_size[1],
        config.domain_height,
    ];
    let mut domain = CartesianCuboid::from_boundaries_and_n_voxels(
        [0.0; 3],
        domain_size,
        [config.n_voxels[0], config.n_voxels[1], 1],
    )
    .map_err(SimulationError::from)?;
    domain.rng_seed = config.rng_seed;
    let domain = CartesianCuboidRods {
        domain,
        gel_pressure: config.gel_pressure,
        surface_friction: config.surface_friction,
        surface_friction_distance: config.surface_friction_distance,
    };

    test_compatibility!(
        aspects: [Mechanics, Interaction, Cycle, DomainForce, NeighborSensing],
        domain: domain,
        agents: agents,
        settings: settings,
    );
    let storage = run_main!(
        agents: agents,
        domain: domain,
        settings: settings,
        aspects: [Mechanics, Interaction, Cycle, DomainForce, NeighborSensing],
        zero_force_default: |c: &RodAgent| {
            nalgebra::MatrixXx3::zeros(c.mechanics.pos().nrows())
        },
        parallelizer: Rayon,
    )?;
    let cells = storage
        .cells
        .load_all_elements()?
        .into_iter()
        .map(|(iteration, cells)| {
            (
                iteration,
                cells
                    .into_iter()
                    .map(|(ident, (cbox, _))| (ident, (cbox.cell, cbox.parent)))
                    .collect(),
            )
        })
        .collect();
    let path = if config.storage_options.contains(&StorageOption::SerdeJson) {
        storage
            .cells
            .extract_builder()
            .get_full_path()
            .parent()
            .map(|x| x.to_path_buf())
    } else {
        None
    };

    Ok(CellContainer::new(cells, path))
}

/// Sorts an iterator of :class:`CellIdentifier` deterministically.
///
/// This function is usefull for generating identical masks every simulation run.
/// This function is implemented as standalone since sorting of a :class:`CellIdentifier` is
/// typically not supported.
///
/// Args:
///     identifiers(list): A list of :class:`CellIdentifier`
///
/// Returns:
///     list: The sorted list.
#[pyfunction]
pub fn sort_cellular_identifiers(
    identifiers: Vec<CellIdentifier>,
) -> Result<Vec<CellIdentifier>, PyErr> {
    let mut identifiers = identifiers;
    identifiers.sort();
    Ok(identifiers)
}

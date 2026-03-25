use std::ops::Deref;

use crate::crm_fit::predict::predict_calculate_cost_rs;

use super::settings::*;

use egobox_doe::SamplingMethod;
use numpy::{PyUntypedArrayMethods, ToPyArray};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Contains information of the progression of the optimization routine
#[pyclass(get_all, set_all)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Final parameters obtained by the optimization
    pub params: Vec<f32>,
    /// Final cost calculated
    pub cost: f32,
    /// If the optimization was successfull
    pub success: Option<bool>,
    /// Number of evaluations
    pub neval: Option<usize>,
    /// Number of total iterations
    pub niter: Option<usize>,
    /// Cost function at each step
    pub evals: Vec<f32>,
}

#[pymethods]
impl OptimizationResult {
    fn save_to_file(&self, filename: std::path::PathBuf) -> PyResult<()> {
        use std::io::prelude::*;
        let output = toml::to_string_pretty(&self)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
        let mut file = std::fs::File::create_new(filename)?;
        file.write_all(output.as_bytes())?;
        Ok(())
    }

    #[staticmethod]
    fn load_from_file(filename: std::path::PathBuf) -> PyResult<Self> {
        let contents = std::fs::read_to_string(filename)?;
        toml::from_str(&contents)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

fn lhs_optimization(
    py: Python,
    n_points: usize,
    n_iter: usize,
    bounds: &numpy::ndarray::Array2<f32>,
    iterations_images: &[usize],
    positions_all: numpy::ndarray::ArrayView4<f32>,
    settings: &Settings,
) -> PyResult<Option<(Vec<f32>, f32, Vec<f32>)>> {
    use kdam::{term::Colorizer, *};
    use rayon::prelude::*;

    let domain_height = settings.domain_height();
    let constants: Constants = settings.constants.extract(py)?;
    let parameter_defs: Parameters = settings.parameters.extract(py)?;
    let config = settings.to_config(py)?;

    let lhs_doe = egobox_doe::Lhs::new(bounds);
    let combinations = lhs_doe.sample(n_points);

    // Initialize progress bar
    kdam::term::init(true);
    let result = kdam::par_tqdm!(
        combinations.axis_iter(ndarray::Axis(0)).into_par_iter(),
        desc = format!("LHS Step {n_iter}").colorize("green"),
        total = n_points
    )
    // Calculate Costs for every sampled parameter point
    .filter_map(|parameters| {
        predict_calculate_cost_rs(
            parameters.to_vec(),
            positions_all,
            domain_height,
            &parameter_defs,
            &constants,
            &config,
            iterations_images,
        )
        .ok()
        .map(|x| (parameters.to_vec(), x, vec![x]))
    })
    .filter(|x| x.1.is_finite())
    .reduce_with(|x, y| if x.1 < y.1 { x } else { y });

    if let Some((_, cost, _)) = result {
        println!("Final cost: {}", format!("{cost}").colorize("blue"));
    }

    Ok(result)
}

fn lhs_optimization_iter(
    py: Python,
    n_points: usize,
    n_steps: usize,
    bounds: &numpy::ndarray::Array2<f32>,
    iterations_images: &[usize],
    positions_all: numpy::ndarray::ArrayView4<f32>,
    settings: &Settings,
    relative_reduction: f32,
) -> PyResult<Option<(Vec<f32>, f32, Vec<f32>)>> {
    // Do lowering via repeated applications of LatinHypercube algorithm
    let mut bounds = bounds.clone();
    let mut result = None;

    for n in 0..n_steps {
        let new_result = lhs_optimization(
            py,
            n_points,
            n,
            &bounds,
            iterations_images,
            positions_all,
            settings,
        )?;
        // If the newly calculated results are better, we update. Otherwise do nothing.
        // This will still decrease the overall size
        if result.is_none() && new_result.is_some() {
            result = new_result;
        } else if let (Some((_, c0, evals_old)), Some((p1, c1, evals_new))) = (&result, new_result)
        {
            if c1 < *c0 {
                let mut evals_combined = evals_old.clone();
                evals_combined.extend(evals_new);
                result = Some((p1, c1, evals_combined));
            }
        }

        if let Some((params, _, _)) = &result {
            let dists = &bounds.column(1) - &bounds.column(0);
            for i in 0..params.len() {
                let d = dists[i];
                let q: f32 = d * relative_reduction / 2.0;
                let b0 = params[i] - q;
                let b1 = params[i] + q;
                bounds[(i, 0)] = bounds[(i, 0)].max(b0);
                bounds[(i, 1)] = bounds[(i, 1)].min(b1);
            }
        } else {
            break;
        }
    }

    Ok(result)
}

/// Performs a full optimization given microscopic data and configuration information of how the
/// parameters should be treated.
#[pyfunction]
#[pyo3(signature = (iterations_images, positions_all, settings, n_workers=-1))]
pub fn run_optimizer(
    py: Python,
    iterations_images: Vec<usize>,
    positions_all: numpy::PyReadonlyArray4<f32>,
    settings: &Settings,
    n_workers: isize,
) -> PyResult<OptimizationResult> {
    env_logger::init();

    let n_agents = positions_all.shape()[1];
    let oinfs = settings.generate_optimization_infos(py, n_agents);
    let OptimizationInfos {
        bounds_lower,
        bounds_upper,
        initial_values,
        ..
    } = oinfs;
    let n_workers = if n_workers <= 0 {
        rayon::max_num_threads()
    } else {
        n_workers as usize
    };

    let bounds = numpy::ndarray::Array2::from_shape_fn((bounds_lower.len(), 2), |(i, j)| {
        if j == 0 {
            bounds_lower[i]
        } else {
            bounds_upper[i]
        }
    });

    let positions_all = positions_all.as_array();

    match settings.optimization.borrow(py).deref() {
        OptimizationMethod::DifferentialEvolution(de) => {
            let locals = pyo3::types::PyDict::new(py);
            let globals = pyo3::types::PyDict::new(py);

            // Required
            locals.set_item("bounds", bounds.to_pyarray(py))?;
            locals.set_item("x0", initial_values.into_pyobject(py)?)?;
            locals.set_item("positions_all", positions_all.to_pyarray(py))?;
            locals.set_item("iterations_images", iterations_images)?;
            locals.set_item("settings", settings.clone().into_pyobject(py)?)?;
            locals.set_item(
                "constraints",
                settings.get_constraints(py, n_agents).into_pyobject(py)?,
            )?;

            // Optional
            locals.set_item("optimization", de.clone().into_pyobject(py)?)?;
            locals.set_item("n_workers", n_workers)?;

            py.run(
                pyo3::ffi::c_str!(
                    r#"
import scipy as sp
import numpy as np
from cr_mech_coli.crm_fit import predict_calculate_cost

args = (positions_all, iterations_images, settings)

evals = []

def callback(intermediate_result):
    fun = intermediate_result.fun
    global evals
    evals.append(float(fun))

A = np.array(constraints)
c = sp.optimize.LinearConstraint(A, lb=0)

res = sp.optimize.differential_evolution(
    predict_calculate_cost,
    bounds=bounds,
    x0=x0,
    args=args,
    workers=n_workers,
    updating="deferred",
    maxiter=optimization.max_iter,
    disp=True,
    tol=optimization.tol,
    recombination=optimization.recombination,
    popsize=optimization.pop_size,
    polish=False,
    rng=optimization.seed,
    callback=callback,
    mutation=optimization.mutation,
    constraints=c,
)

if optimization.polish is not None:
    res = sp.optimize.minimize(
        predict_calculate_cost,
        bounds=bounds,
        x0=res.x,
        args=args,
        method=optimization.polish.method,
        options={
            "disp": True,
            "maxiter": optimization.polish.max_iter,
        },
        constraints=c,
    )
"#
                ),
                Some(&globals),
                Some(&locals),
            )?;
            let res = locals.get_item("res")?.unwrap();
            let params: Vec<f32> = res.get_item("x")?.extract()?;
            let cost: f32 = res.get_item("fun")?.extract()?;
            let success: Option<bool> = res.get_item("success").ok().and_then(|x| x.extract().ok());
            let neval: Option<usize> = res.get_item("nfev").ok().and_then(|x| x.extract().ok());
            let niter: Option<usize> = res.get_item("nit").ok().and_then(|x| x.extract().ok());
            let evals: Vec<f32> = locals
                .get_item("evals")?
                .and_then(|x| x.extract().ok())
                .unwrap_or(
                    globals
                        .get_item("evals")?
                        .and_then(|x| x.extract().ok())
                        .unwrap(),
                );

            Ok(OptimizationResult {
                params,
                cost,
                success,
                neval,
                niter,
                evals,
            })
        }
        OptimizationMethod::LatinHypercube(lhs) => {
            // Sample the space
            let LatinHypercube {
                n_points,
                n_steps,
                relative_reduction,
            } = lhs;

            // Initialize threadpool with correct number of workers
            rayon::ThreadPoolBuilder::new()
                .num_threads(n_workers)
                .build_global()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;

            let result = lhs_optimization_iter(
                py,
                *n_points,
                *n_steps,
                &bounds,
                &iterations_images,
                positions_all,
                settings,
                *relative_reduction,
            )?;

            if let Some((params, cost, evals)) = result {
                Ok(OptimizationResult {
                    params,
                    cost,
                    success: Some(true),
                    neval: Some(n_steps * n_points),
                    niter: Some(*n_steps),
                    evals,
                })
            } else {
                Ok(OptimizationResult {
                    params: initial_values,
                    cost: f32::INFINITY,
                    success: Some(false),
                    neval: None,
                    niter: None,
                    evals: vec![],
                })
            }
        }
        OptimizationMethod::LHSNelderMead(nelder_mead) => {
            let LHSNelderMead {
                max_iter,
                latin_hypercube,
            } = nelder_mead;
            let result = if let Some(LatinHypercube {
                n_points,
                n_steps,
                relative_reduction,
            }) = latin_hypercube
            {
                lhs_optimization_iter(
                    py,
                    *n_points,
                    *n_steps,
                    &bounds,
                    &iterations_images,
                    positions_all,
                    settings,
                    *relative_reduction,
                )?
            } else {
                Some((initial_values.clone(), f32::MAX, vec![]))
            };

            let params = if let Some((params, _, _)) = result {
                params
            } else {
                return Ok(OptimizationResult {
                    params: initial_values,
                    cost: f32::MAX,
                    success: Some(false),
                    neval: None,
                    niter: None,
                    evals: vec![],
                });
            };

            // Loal Minimization at end
            // Required
            let globals = pyo3::types::PyDict::new(py);
            let locals = pyo3::types::PyDict::new(py);
            locals.set_item("bounds", bounds.to_pyarray(py))?;
            locals.set_item("x0", params.into_pyobject(py)?)?;
            locals.set_item("positions_all", positions_all.to_pyarray(py))?;
            locals.set_item("iterations", iterations_images)?;
            locals.set_item("settings", settings.clone().into_pyobject(py)?)?;
            locals.set_item("disp", true)?;

            // Optional
            locals.set_item("max_iter", max_iter)?;

            py.run(
                pyo3::ffi::c_str!(
                    r#"
import scipy as sp
from cr_mech_coli.crm_fit import predict_calculate_cost

args = (positions_all, iterations, settings)

evals = []

def callback(intermediate_result):
    # nit = intermediate_result.nit
    fun = intermediate_result.fun
    global evals
    evals.append(fun)
    print(f"Objective Function: {fun}")

print("Starting Nelder-Mead Optimization")
res = sp.optimize.minimize(
    predict_calculate_cost,
    x0=x0,
    args=args,
    bounds=bounds,
    method="Nelder-Mead",
    options={
        "disp": disp,
        "maxiter": max_iter,
        "maxfev": max_iter,
    },
    callback=callback,
)
            "#
                ),
                Some(&globals),
                Some(&locals),
            )?;
            let res = locals.get_item("res")?.unwrap();
            let params: Vec<f32> = res.get_item("x")?.extract()?;
            let cost: f32 = res.get_item("fun")?.extract()?;
            let success: Option<bool> = res.get_item("success").ok().and_then(|x| x.extract().ok());
            let neval: usize = res
                .get_item("nfev")
                .ok()
                .and_then(|x| x.extract().ok())
                .unwrap_or(0);
            let niter = res.get_item("nit").ok().and_then(|x| x.extract().ok());
            let evals: Vec<f32> = locals
                .get_item("evals")?
                .and_then(|x| x.extract().ok())
                .unwrap_or(
                    globals
                        .get_item("evals")?
                        .and_then(|x| x.extract().ok())
                        .unwrap(),
                );

            Ok(OptimizationResult {
                params,
                cost,
                success,
                neval: Some(neval),
                niter,
                evals,
            })
        }
    }
}

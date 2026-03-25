use core::f32;
use std::ops::Deref;

use approx::AbsDiffEq;
use cellular_raza::prelude::StorageOption;
use pyo3::{prelude::*, IntoPyObjectExt};
use serde::{Deserialize, Serialize};

/// Float which is sampled by the optimizer
#[pyclass(get_all, set_all, module = "cr_mech_coli.crm_fit")]
#[derive(Clone, Debug, Serialize, Deserialize, AbsDiffEq, PartialEq)]
#[approx(epsilon_type = f32)]
pub struct SampledFloat {
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Initial value (if applicable for the optimizer)
    pub initial: f32,
    /// Determines if the parameter should be treated as given per cell or if it is identical
    /// across all cells.
    #[approx(equal)]
    pub individual: Option<bool>,
}

#[pymethods]
impl SampledFloat {
    #[new]
    #[pyo3(signature = (min, max, initial, individual=false))]
    fn new(min: f32, max: f32, initial: f32, individual: Option<bool>) -> Self {
        Self {
            min,
            max,
            initial,
            individual,
        }
    }
}

/// This enum has 3 variants:
///
/// - :class:`SampledFloat` Samples the value in the given range
/// - :class:`float` Fixes it to the given value
/// - :class:`list` Fixes it on a per-agent basis to the given values.
#[pyclass(get_all, set_all, module = "cr_mech_coli.crm_fit")]
#[derive(Clone, Debug, Serialize, Deserialize, AbsDiffEq, PartialEq)]
#[approx(epsilon_type = f32)]
pub enum Parameter {
    /// Contains a :class:`.SampledFloat`
    #[serde(untagged)]
    SampledFloat(SampledFloat),
    /// Contains a fixed value for the parameter
    #[serde(untagged)]
    Float(f32),
    /// Contains a list of fixed values which are applied to the individual cells.
    #[serde(untagged)]
    #[approx(into_iter)]
    List(Vec<f32>),
}

#[pymethods]
impl Parameter {
    fn __repr__(&self) -> String {
        use Parameter::*;
        match self {
            SampledFloat(s) => format!("{:#?}", s),
            Float(f) => format!("{:#?}", f),
            List(l) => format!("{:?}", l),
        }
    }

    /// Obtains the inner value of the enum
    ///
    /// This will cast to a dict, list or float.
    pub fn get_inner(&self, py: Python) -> PyResult<Py<PyAny>> {
        use Parameter::*;
        match self {
            SampledFloat(s) => s.clone().into_py_any(py),
            Float(f) => f.into_py_any(py),
            List(l) => l.into_py_any(py),
        }
    }
}

fn parameter_from_obj(obj: &Bound<PyAny>) -> PyResult<Parameter> {
    if let Ok(value) = obj.extract() {
        Ok(Parameter::Float(value))
    } else if let Ok(value) = obj.extract() {
        Ok(Parameter::SampledFloat(value))
    } else if let Ok(value) = obj.extract() {
        Ok(Parameter::List(value))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Cannot convert object to SampledFloat",
        ))
    }
}

/// Contains all parameters for the `crm_fit` script
#[pyclass(get_all, module = "cr_mech_coli.crm_fit")]
#[derive(Clone, Debug, Serialize, Deserialize, AbsDiffEq, PartialEq)]
pub struct Parameters {
    /// Thickness of the bacterial rod
    pub radius: Parameter,
    /// Rigidity which controls the mechanical flexibility of the rod
    pub rigidity: Parameter,
    /// Auxiliary parameter to enforce the correct distance of vertices
    pub spring_tension: Parameter,
    /// Stokesian damping constant
    pub damping: Parameter,
    /// Interaction strength of the potential
    pub strength: Parameter,
    /// The type of interaction potential. See :class:`.PotentialType`.
    pub potential_type: PotentialType,
    /// Growth rate of the bacterium
    pub growth_rate: Parameter,
}

macro_rules! impl_setters(
    (@single $struct_name:ident $name:ident $setter:ident $counter:ident) => {
        #[pymethods]
        impl $struct_name {
            fn $setter (&mut self, obj: &Bound<PyAny>, n_agents: usize) -> PyResult<(usize, usize)> {
                let param = parameter_from_obj(obj)?;
                let n_params = param_counter!(self, $name, n_agents);
                let n_before = self. $counter(n_agents);

                self.$name = param;
                Ok((n_before, n_params))
            }
        }
    };
    ($struct_name:ident; $($name:ident $setter:ident $counter:ident;)*) => {
        $(impl_setters!{@single $struct_name $name $setter $counter})*
    };
);

macro_rules! param_counter(
    ($self:ident, $pname:ident, $n_agents:ident) => {
        match &$self . $pname {
            Parameter::SampledFloat(sf) => {
                if sf.individual == Some(true) {
                    $n_agents
                } else {
                    1
                }
            }
            Parameter::Float(_) => 0,
            Parameter::List(_) => 0,
        }
    };
);

macro_rules! impl_count_before(
    ($counter_name:ident $(,$before:ident)*) => {
        #[pymethods]
        impl Parameters {
            #[allow(unused)]
            fn $counter_name (&self, n_agents: usize) -> usize {
                0 $( + param_counter!(self, $before, n_agents))*
            }
        }
    };
    (@pot $counter_name:ident, $pot:ident $(,$before:ident)*) => {
        /// Sets this variable
        ///
        /// This also returns the number of optimizable parameters before and how many parameters
        /// would be changed.
        /// This allows adjusting a vector of parameters such as in the :class`OptimizationResult`
        #[pymethods]
        impl Parameters {
            #[allow(unused)]
            fn $counter_name (&self, n_agents: usize) -> usize {
                let mut counter = 0 $( + param_counter!(self, $before, n_agents))*;
                counter += match &self.potential_type {
                    PotentialType::Mie(mie) => {0
                        + param_counter!(mie, en, n_agents)
                        + param_counter!(mie, em, n_agents)
                    },
                    PotentialType::Morse(morse) => {
                        param_counter!(morse, potential_stiffness, n_agents)
                    }
                };
                counter
            }
        }
    };
);

impl_count_before!(count_radius);
impl_count_before!(count_rigidity, radius);
impl_count_before!(count_spring_tension, radius, rigidity);
impl_count_before!(count_damping, radius, rigidity, spring_tension);
impl_count_before!(count_strength, radius, rigidity, spring_tension, damping);
impl_count_before!(
    count_growth_rate,
    radius,
    rigidity,
    spring_tension,
    damping,
    strength
);
impl_count_before!(
    count_potential_type,
    radius,
    rigidity,
    spring_tension,
    damping,
    strength,
    growth_rate
);

impl_setters!(
    Parameters;
    radius set_radius count_radius;
    rigidity set_rigidity count_rigidity;
    spring_tension set_spring_tension count_spring_tension;
    damping set_damping count_damping;
    strength set_strength count_strength;
    growth_rate set_growth_rate count_growth_rate;
);

/// Parametes for the :class:`MorsePotentialF32`
#[pyclass(get_all, set_all, module = "cr_mech_coli.crm_fit")]
#[derive(Clone, Debug, Serialize, Deserialize, AbsDiffEq, PartialEq)]
#[approx(epsilon_type = f32)]
pub struct Morse {
    /// Potential stiffness
    pub potential_stiffness: Parameter,
}

/// Parameters for the :class:`.MiePotentialF32`
#[pyclass(get_all, set_all, module = "cr_mech_coli.crm_fit")]
#[derive(Clone, Debug, Serialize, Deserialize, AbsDiffEq, PartialEq)]
#[approx(epsilon_type = f32)]
pub struct Mie {
    /// Exponent `n`
    pub en: Parameter,
    /// Exponent `m`
    pub em: Parameter,
    /// Upper bound of the potential force
    pub bound: f32,
}

/// The interaction type provided to the optimization routine
///
/// See :class:`.Mie` and :class:`.Morse`.
#[pyclass(get_all, set_all, module = "cr_mech_coli.crm_fit")]
#[derive(Clone, Debug, Serialize, Deserialize, AbsDiffEq, PartialEq)]
pub enum PotentialType {
    /// See :class:`.Mie`
    Mie(Mie),
    /// See :class:`.Morse`
    Morse(Morse),
}

#[pymethods]
impl PotentialType {
    // Reconstructs a interaction potential
    // pub fn reconstruct_potential(&self, radius: f32, strength: f32, cutoff: f32) {}

    /// Formats the object
    pub fn to_short_string(&self) -> String {
        match self {
            PotentialType::Mie(_) => "mie".to_string(),
            PotentialType::Morse(_) => "morse".to_string(),
        }
    }

    /// Helper method for :func:`~PotentialType.__reduce__`
    #[staticmethod]
    fn deserialize(data: Vec<u8>) -> Self {
        serde_pickle::from_slice(&data, Default::default()).unwrap()
    }

    /// Used to pickle the :class:`PotentialType`
    fn __reduce__<'py>(
        &'py self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        py.run(
            &std::ffi::CString::new("from cr_mech_coli.crm_fit.crm_fit_rs import PotentialType")?,
            None,
            None,
        )
        .unwrap();
        let deserialize = py.eval(
            &std::ffi::CString::new("PotentialType.deserialize")?,
            None,
            None,
        )?;
        let data = serde_pickle::to_vec(&self, Default::default()).unwrap();
        Ok((
            deserialize.into_pyobject_or_pyerr(py)?.into_any(),
            (data,).into_pyobject_or_pyerr(py)?.into_any(),
        ))
    }

    /// Obtains an copy of the inner type for Mie or Morse potential
    pub fn clone_inner<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            PotentialType::Mie(mie) => mie.clone().into_bound_py_any(py),
            PotentialType::Morse(morse) => morse.clone().into_bound_py_any(py),
        }
    }
}

/// Polishing options for the :class:`DifferentialEvolution` method
#[pyclass(get_all, set_all, module = "cr_mech_coli.crm_fit")]
#[derive(Clone, Debug, Serialize, Deserialize, AbsDiffEq, PartialEq)]
#[approx(epsilon_type = f32)]
pub struct Polish {
    #[serde(default = "default_polish_method")]
    #[approx(equal)]
    method: String,
    #[serde(default = "default_polish_max_iter")]
    #[approx(equal)]
    max_iter: usize,
}

fn default_polish_method() -> String {
    "COBYQA".to_string()
}

fn default_polish_max_iter() -> usize {
    100
}

/// Parameters to control optimization scheme via the `differential_evolution` algorithm.
#[pyclass(get_all, set_all, module = "cr_mech_coli.crm_fit")]
#[derive(Clone, Debug, Serialize, Deserialize, AbsDiffEq, PartialEq)]
#[approx(epsilon_type = f32)]
pub struct DifferentialEvolution {
    /// Initial seed of the differential evolution algorithm
    #[serde(default)]
    #[approx(equal)]
    pub seed: u64,
    /// Tolerance of the differential evolution algorithm
    #[serde(default = "default_tol")]
    pub tol: f32,
    /// Maximum iterations of the differential evolution algorithm
    #[serde(default = "default_max_iter")]
    #[approx(equal)]
    pub max_iter: usize,
    /// Population size for each iteration
    #[serde(default = "default_pop_size")]
    #[approx(equal)]
    pub pop_size: usize,
    /// Recombination value of the differential evolution algorithm
    #[serde(default = "default_recombination")]
    pub recombination: f32,
    /// Mutation variable of the differential evolution algorithm
    #[approx(epsilon_map = |x| (x, x))]
    #[serde(default = "default_mutation")]
    pub mutation: (f32, f32),
    /// Determines if the final result should be polished
    // #[approx(equal)]
    // #[serde(default = "default_polish")]
    // pub polish: bool,
    #[approx(equal)]
    #[serde(default)]
    pub polish: Option<Polish>,
}

/// Parameters to control optimization scheme via an incremental lowering with the Latin-Hypercube algorithm.
#[pyclass(get_all, set_all, module = "cr_mech_coli.crm_fit")]
#[derive(Clone, Debug, Serialize, Deserialize, AbsDiffEq, PartialEq)]
#[approx(epsilon_type = f32)]
pub struct LatinHypercube {
    /// Number of points which should be sampled
    #[approx(equal)]
    pub n_points: usize,
    /// Maximum lowering steps of the iterative approach
    #[approx(equal)]
    #[serde(default = "Default::default")]
    pub n_steps: usize,
    /// Relative reduction of bound size per lowering step
    pub relative_reduction: f32,
}

/// Utilizes the :class:`.LatinHypercube` and performs a local minimization with the Nelder-Mead
/// method of `scipy`.
#[pyclass(get_all, set_all, module = "cr_mech_coli.crm_fit")]
#[derive(Clone, Debug, Serialize, Deserialize, AbsDiffEq, PartialEq)]
#[approx(epsilon_type = f32)]
pub struct LHSNelderMead {
    /// Maximum number of iterations
    #[approx(equal)]
    pub max_iter: usize,
    /// Performs a LatinHypercube sweep before applying the Nelder-Mead method
    #[serde(rename = "latin_hypercube")]
    pub latin_hypercube: Option<LatinHypercube>,
}

/// Other settings which are not related to the outcome of the simulation
#[pyclass(get_all, set_all, module = "cr_mech_coli.crm_fit")]
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Others {
    /// Show/hide progressbar for solving of one single simulation
    pub progressbar: Option<String>,
}

#[pymethods]
impl Others {
    #[new]
    #[pyo3(signature = (progressbar=None))]
    fn new(progressbar: Option<String>) -> Self {
        Others { progressbar }
    }
}

pub(crate) const fn default_tol() -> f32 {
    1e-4
}

pub(crate) const fn default_max_iter() -> usize {
    50
}

pub(crate) const fn default_pop_size() -> usize {
    100
}

pub(crate) const fn default_recombination() -> f32 {
    0.3
}

pub(crate) const fn default_mutation() -> (f32, f32) {
    (0.5, 1.5)
}

/// Contains all constants of the numerical simulation
#[pyclass(get_all, set_all, module = "cr_mech_coli.crm_fit")]
#[derive(Clone, Debug, Serialize, Deserialize, AbsDiffEq, PartialEq)]
pub struct Constants {
    /// Total time from start to finish
    pub t_max: f32,
    /// Time increment used to solve equations
    pub dt: f32,
    /// Size of the domain
    #[approx(into_iter)]
    pub domain_size: [f32; 2],
    /// Number of voxels to dissect the domain into
    #[approx(equal)]
    #[serde(default = "default_n_voxels")]
    pub n_voxels: [core::num::NonZeroUsize; 2],
    /// Random initial seed
    #[approx(equal)]
    pub rng_seed: u64,
    /// Cutoff after which the physical interaction is identically zero
    pub cutoff: f32,
    /// Number of vertices to use for discretization of agents
    #[approx(equal)]
    pub n_vertices: core::num::NonZeroUsize,
    /// Number of save points which are not initial and final time point
    #[approx(equal)]
    #[serde(default = "default_n_saves")]
    pub n_saves: usize,
    /// Error used to draw profiles
    pub displacement_error: f32,
    #[serde(default = "default_error_cost")]
    /// Cost which is assigned when the produced value is not normal i.e. NaN, +/- infinity
    pub error_cost: f32,
}

const fn default_n_voxels() -> [core::num::NonZeroUsize; 2] {
    [unsafe { core::num::NonZeroUsize::new_unchecked(1) }; 2]
}

const fn default_n_saves() -> usize {
    0
}

const fn default_error_cost() -> f32 {
    1e9
}

pub(crate) fn get_inner<T>(ptp: &Py<T>, py: Python) -> T
where
    T: for<'a, 'py> pyo3::conversion::FromPyObject<'a, 'py>,
    for<'a, 'py> <T as pyo3::conversion::FromPyObject<'a, 'py>>::Error: std::fmt::Debug,
{
    ptp.extract(py).unwrap()
}

/// Contains settings for the various optimization routines.
#[pyclass(module = "cr_mech_coli.crm_fit")]
#[derive(Clone, Debug, Serialize, Deserialize, AbsDiffEq, PartialEq)]
pub enum OptimizationMethod {
    /// Settings for the :class:`Optimization` method.
    #[serde(rename = "differential_evolution")]
    DifferentialEvolution(DifferentialEvolution),
    /// Settings for the :class:`LatinHypercube` method.
    #[serde(rename = "latin_hypercube")]
    LatinHypercube(LatinHypercube),
    /// Uses [egobox_ego] to optimize
    #[serde(rename = "lhs_nelder_mead")]
    LHSNelderMead(LHSNelderMead),
}

/// Return type of the :meth:`Settings.generate_optimization_infos` method.
#[pyclass(get_all, set_all)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizationInfos {
    /// Lower Bounds
    pub bounds_lower: Vec<f32>,
    /// Upper Bounds
    pub bounds_upper: Vec<f32>,
    /// Initial Guess for parameters
    pub initial_values: Vec<f32>,
    /// Information generated for parameters
    pub parameter_infos: Vec<(String, String, String)>,
    /// Values of constants
    pub constants: Vec<f32>,
    /// Information for constants
    pub constant_infos: Vec<(String, String, String)>,
}

/// Contains all settings required to fit the model to images
#[pyclass(get_all, set_all, module = "cr_mech_coli.crm_fit")]
#[derive(Clone, Debug, Serialize, Deserialize, AbsDiffEq)]
#[approx(epsilon_type = f32)]
pub struct Settings {
    /// See :class:`Constants`
    #[approx(map = |b| Python::attach(|py| Some(get_inner(b, py))))]
    pub constants: Py<Constants>,
    /// See :class:`Parameters`
    #[approx(map = |b| Python::attach(|py| Some(get_inner(b, py))))]
    pub parameters: Py<Parameters>,
    /// See :class:`OptimizationMethod`
    #[approx(map = |b| Python::attach(|py| Some(get_inner(b, py))))]
    pub optimization: Py<OptimizationMethod>,
    /// See :class:`Others`
    #[approx(skip)]
    pub others: Option<Py<Others>>,
}

impl PartialEq for Settings {
    fn eq(&self, other: &Self) -> bool {
        let Self {
            constants,
            parameters,
            optimization,
            others,
        } = &self;
        Python::attach(|py| {
            constants.borrow(py).eq(&other.constants.borrow(py))
                && parameters.borrow(py).eq(&other.parameters.borrow(py))
                && optimization.borrow(py).eq(&other.optimization.borrow(py))
                && if let (Some(s), Some(o)) = (&others, &other.others) {
                    s.borrow(py).eq(&o.borrow(py))
                } else {
                    true
                }
        })
    }
}

#[pymethods]
impl Settings {
    /// Creates a :class:`Settings` from a given toml string.
    /// See also :func:`~Settings.from_toml_string`.
    #[staticmethod]
    pub fn from_toml(toml_filename: std::path::PathBuf) -> PyResult<Self> {
        let content = std::fs::read_to_string(toml_filename)?;
        Self::from_toml_string(&content)
    }

    /// Parses the contents of the given string and returns a :class:`Settings` object.
    /// See also :func:`~Settings.from_toml`.
    #[staticmethod]
    pub fn from_toml_string(toml_string: &str) -> PyResult<Self> {
        toml::from_str(toml_string)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }

    /// Creates a toml string from the configuration file
    pub fn to_toml(&self) -> PyResult<String> {
        toml::to_string(&self).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }

    /// Obtains the domain height
    #[getter]
    pub fn domain_height(&self) -> f32 {
        2.5
    }

    /// Helper method for :func:`~PotentialType.__reduce__`
    #[staticmethod]
    fn deserialize(data: Vec<u8>) -> Self {
        serde_pickle::from_slice(&data, Default::default()).unwrap()
    }

    /// Implements the `__reduce__` method used by pythons pickle protocol.
    pub fn __reduce__<'py>(
        &'py self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        py.run(
            &std::ffi::CString::new("from cr_mech_coli.crm_fit.crm_fit_rs import Settings")?,
            None,
            None,
        )?;
        // py.run_bound("from crm_fit import deserialize_potential_type", None, None)
        //     .unwrap();
        let deserialize = py.eval(&std::ffi::CString::new("Settings.deserialize")?, None, None)?;
        let data = serde_pickle::to_vec(&self, Default::default()).unwrap();
        Ok((
            deserialize.into_pyobject_or_pyerr(py)?.into_any(),
            (data,).into_pyobject_or_pyerr(py)?.into_any(),
        ))
    }

    /// Get constraints for the optimization problem
    pub fn get_constraints(&self, py: Python, n_agents: usize) -> Vec<f32> {
        let mut constraints = Vec::new();
        let parameters = self.parameters.borrow(py);
        macro_rules! push_if_individual(
            ($param:ident) => {
                if let Parameter::SampledFloat(s) = &parameters.$param {
                    if s.individual.is_some_and(|x| x) {
                        constraints.extend(vec![0.0; n_agents]);
                    } else {
                        constraints.push(0.0);
                    }
                }
            };
        );
        push_if_individual!(radius);
        push_if_individual!(rigidity);
        push_if_individual!(spring_tension);
        push_if_individual!(damping);
        push_if_individual!(strength);
        push_if_individual!(growth_rate);
        if let PotentialType::Mie(Mie {
            en: Parameter::SampledFloat(s1),
            em: Parameter::SampledFloat(s2),
            ..
        }) = &parameters.potential_type
        {
            if s1.individual.is_some_and(|x| x) && s2.individual.is_some_and(|x| x) {
                todo!()
            } else {
                constraints.push(1.0);
                constraints.push(-1.0);
            }
        }

        constraints
    }

    /// Converts the settings provided to a :class:`Configuration` object required to run the
    /// simulation
    pub fn to_config(&self, py: Python) -> PyResult<crate::Configuration> {
        #[allow(unused)]
        let Self {
            constants,
            parameters,
            optimization,
            others,
        } = self.clone();
        let Constants {
            t_max,
            dt,
            domain_size,
            n_voxels,
            rng_seed,
            cutoff: _,
            n_vertices: _,
            n_saves,
            displacement_error: _,
            error_cost: _,
        } = constants.extract(py)?;
        let Others { progressbar } = if let Some(o) = others {
            o.borrow(py).deref().clone()
        } else {
            Others::default()
        };
        Ok(crate::Configuration {
            domain_height: self.domain_height(),
            n_threads: 1.try_into().unwrap(),
            t0: 0.0,
            dt,
            t_max,
            n_saves,
            progressbar,
            domain_size,
            n_voxels: [n_voxels[0].get(), n_voxels[1].get()],
            rng_seed,
            gel_pressure: 0.,
            surface_friction: 0.,
            surface_friction_distance: 1.,
            storage_options: vec![StorageOption::Memory],
            storage_location: std::path::PathBuf::new(),
            storage_suffix: None,
        })
    }

    /// Creates a list of lower and upper bounds for the sampled parameters
    #[allow(unused)]
    pub fn generate_optimization_infos(&self, py: Python, n_agents: usize) -> OptimizationInfos {
        let mut param_space_dim = 0;

        #[allow(unused)]
        let Parameters {
            radius,
            rigidity,
            spring_tension,
            damping,
            strength,
            potential_type,
            growth_rate,
        } = &self.parameters.borrow(py).deref().clone();

        let mut bounds_lower = Vec::new();
        let mut bounds_upper = Vec::new();
        let mut initial_values = Vec::new();
        let mut infos = Vec::new();
        let mut constants = Vec::new();
        let mut constant_infos = Vec::new();
        macro_rules! append_infos_bounds(
            ($var:expr, $var_name:expr, $units:expr, $symbol:expr) => {
                match &$var {
                    Parameter::SampledFloat(SampledFloat {
                        min,
                        max,
                        initial,
                        individual,
                    }) => {
                        if individual.is_none() || individual == &Some(false) {
                            bounds_lower.push(min.clone());
                            bounds_upper.push(max.clone());
                            param_space_dim += 1;
                            infos.push((
                                $var_name.to_string(),
                                $units.to_string(),
                                $symbol.to_string(),
                            ));
                            initial_values.push(initial.clone());
                        } else {
                            bounds_lower.extend(vec![min.clone(); n_agents]);
                            bounds_upper.extend(vec![max.clone(); n_agents]);
                            param_space_dim += n_agents;
                            infos.extend(
                                (0..n_agents)
                                    .map(|i| (
                                        format!("{} {}", $var_name, i),
                                        $units.to_string(),
                                        format!("{}", $symbol),
                                    ))
                            );
                            initial_values.extend(vec![initial.clone(); n_agents]);
                        }
                    },
                    Parameter::Float(c) => {
                        constants.push(*c);
                        constant_infos.push((
                            $var_name.to_string(),
                            $units.to_string(),
                            $symbol.to_string(),
                        ));
                    },
                    Parameter::List(list) => {
                        constants.extend(list);
                        constant_infos.push((
                            $var_name.to_string(),
                            $units.to_string(),
                            $symbol.to_string(),
                        ));
                    },
                }
            }
        );
        append_infos_bounds!(radius, "Radius", "µm", "r");
        append_infos_bounds!(rigidity, "Rigidity", "µm/min", "κ");
        append_infos_bounds!(spring_tension, "Spring Tension", "1/min²", "γ");
        append_infos_bounds!(damping, "Damping", "1/min", "λ");
        append_infos_bounds!(strength, "Strength", "µm^2/min^2", "V$_0$");
        append_infos_bounds!(growth_rate, "Growth Rate", "1/min", "µ");
        match potential_type {
            PotentialType::Mie(mie) => {
                let en = mie.en.clone();
                let em = mie.em.clone();
                append_infos_bounds!(en, "Exponent n", "1", "n");
                append_infos_bounds!(em, "Exponent m", "1", "m");
            }
            PotentialType::Morse(morse) => {
                append_infos_bounds!(&morse.potential_stiffness, "Potential Stiffness", "µm", "λ")
            }
        }

        OptimizationInfos {
            bounds_lower,
            bounds_upper,
            initial_values,
            parameter_infos: infos,
            constants,
            constant_infos,
        }
    }

    /// Obtains all values for individual parameters
    pub fn get_parameters_distributions(
        &self,
        py: Python,
        n_agents: usize,
        optizmization_result: &super::optimize::OptimizationResult,
    ) -> Vec<(usize, String, Vec<f32>)> {
        let _b = self.parameters.borrow(py);
        let Parameters {
            radius,
            rigidity,
            spring_tension,
            damping,
            strength,
            #[allow(unused)]
            potential_type,
            growth_rate,
        } = _b.deref();

        let mut counter = 0;
        let mut values = vec![];
        macro_rules! append_if_individual(
            ($var:expr, $var_name:expr) => {
                #[allow(unused)]
                match $var {
                    Parameter::SampledFloat(SampledFloat {
                        min: _,
                        max: _,
                        initial: _,
                        individual: Some(true),
                    }) => {
                        let v = (0..n_agents)
                            .map(|i| self.get_param(
                                py,
                                &$var_name.to_lowercase(),
                                optizmization_result,
                                n_agents,
                                i
                            ))
                            .collect::<Vec<_>>();
                        values.push((counter, $var_name.to_string(), v));
                        counter += n_agents;
                    },
                    Parameter::SampledFloat(SampledFloat {
                        min: _,
                        max: _,
                        initial: _,
                        individual,
                    }) => counter += 1,
                    _ => (),
                }
            }
        );

        append_if_individual!(radius, "Radius");
        append_if_individual!(rigidity, "Rigidity");
        append_if_individual!(spring_tension, "Spring Tension");
        append_if_individual!(damping, "Damping");
        append_if_individual!(strength, "Strength");
        append_if_individual!(growth_rate, "Growth Rate");

        values
    }

    /// Formats the object
    pub fn __repr__(&self) -> String {
        format!("{self:#?}")
    }

    /// Return a parameter which has been obtained either by optimization or fixed initially.
    pub fn get_param(
        &self,
        py: Python,
        param_name: &str,
        optizmization_result: &super::optimize::OptimizationResult,
        n_agents: usize,
        agent_index: usize,
    ) -> f32 {
        // Check if the parameter used for optimization contain the queried parameter name
        let param_name_cleaned = param_name.trim().to_lowercase();
        let parameter_infos = self
            .generate_optimization_infos(py, n_agents)
            .parameter_infos;

        let mut first_last = None;
        for (n, (pname, _, _)) in parameter_infos.iter().enumerate() {
            // This is a hack since we append parameter names with a number corresponding to the
            // cell. Thus we check if is contained rather than identical.
            if pname.trim().to_lowercase().contains(&param_name_cleaned) {
                if first_last.is_none() {
                    first_last = Some((n, n));
                }
                if let Some((m, _)) = first_last {
                    first_last = Some((m, n));
                }
            }
        }
        if let Some((n, m)) = first_last {
            if n == m {
                return optizmization_result.params[n];
            } else {
                return optizmization_result.params[n + agent_index];
            }
        }

        use PotentialType::*;
        macro_rules! find_param(
            (@interaction $potential_type:ident, $param_str:literal, $param_field:ident) => {
                if param_name_cleaned == $param_str.trim().to_lowercase() {
                    if let $potential_type(pot) = &self.parameters.bind(py).borrow().potential_type
                    {
                        match &pot.$param_field {
                            Parameter::Float(value) => return *value,
                            Parameter::List(list) => return list[agent_index],
                            _ => (),
                        }
                    }
                }
            };
            ($param_str:literal, $param_field:ident) => {
                if param_name_cleaned == $param_str.trim().to_lowercase() {
                    match &self.parameters.bind(py).borrow().$param_field {
                        Parameter::Float(value) => return *value,
                        Parameter::List(list) => return list[agent_index],
                        _ => (),
                    }
                }
            };
        );

        find_param!("Radius", radius);
        find_param!("Rigidity", rigidity);
        find_param!("Spring Tension", spring_tension);
        find_param!("Damping", damping);
        find_param!("Strength", strength);
        find_param!("Growth Rate", growth_rate);
        find_param!(@interaction Mie, "Exponent n", en);
        find_param!(@interaction Mie, "Exponent m", em);
        if param_name_cleaned == "bound" {
            if let Mie(pot) = &self.parameters.bind(py).borrow().potential_type {
                return pot.bound;
            }
        }
        find_param!(@interaction Morse, "Potential Stiffness", potential_stiffness);

        panic!(
            "Parameter with name {param_name} at agent index {agent_index} could not be obtained"
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn generate_test_settings() -> PyResult<(Settings, String)> {
        Python::initialize();
        Python::attach(|py| -> PyResult<(Settings, String)> {
            let potential_type = PotentialType::Mie(Mie {
                en: Parameter::SampledFloat(SampledFloat {
                    min: 0.2,
                    max: 25.0,
                    initial: 6.0,
                    individual: Some(false),
                }),
                em: Parameter::SampledFloat(SampledFloat {
                    min: 0.2,
                    max: 25.0,
                    initial: 5.5,
                    individual: None,
                }),
                bound: 8.0,
            });
            let settings1 = Settings {
                constants: Py::new(
                    py,
                    Constants {
                        t_max: 100.0,
                        dt: 0.005,
                        domain_size: [100.0; 2],
                        n_voxels: [1.try_into().unwrap(); 2],
                        rng_seed: 0,
                        cutoff: 20.0,
                        n_vertices: 8.try_into().unwrap(),
                        n_saves: 0,
                        displacement_error: 0.5,
                        error_cost: default_error_cost(),
                    },
                )?,
                parameters: Py::new(
                    py,
                    Parameters {
                        radius: Parameter::SampledFloat(SampledFloat {
                            min: 3.0,
                            max: 6.0,
                            initial: 4.5,
                            individual: Some(true),
                        }),
                        rigidity: Parameter::Float(8.0),
                        spring_tension: Parameter::Float(1.0),
                        damping: Parameter::SampledFloat(SampledFloat {
                            min: 0.6,
                            max: 2.5,
                            initial: 1.5,
                            individual: None,
                        }),
                        strength: Parameter::SampledFloat(SampledFloat {
                            min: 1.0,
                            max: 4.5,
                            initial: 1.0,
                            individual: None,
                        }),
                        potential_type,
                        growth_rate: Parameter::SampledFloat(SampledFloat {
                            min: 0.0,
                            max: 10.0,
                            initial: 1.0,
                            individual: None,
                        }),
                    },
                )?,
                optimization: Py::new(
                    py,
                    OptimizationMethod::DifferentialEvolution(DifferentialEvolution {
                        seed: 0,
                        tol: 1e-3,
                        max_iter: default_max_iter(),
                        pop_size: default_pop_size(),
                        recombination: default_recombination(),
                        mutation: default_mutation(),
                        polish: Some(Polish {
                            method: "HansWurst".to_string(),
                            max_iter: 122,
                        }),
                    }),
                )?,
                others: Some(Py::new(py, Others { progressbar: None })?),
            };
            let toml_string = "
[constants]
t_max=100.0
dt=0.005
domain_size=[100, 100]
n_voxels=[1, 1]
rng_seed=0
cutoff=20.0
n_vertices=8
displacement_error=0.5

[parameters]
radius = { min = 3.0, max=6.0, initial=4.5, individual=true }
rigidity = 8.0
spring_tension = 1.0
damping = { min=0.6, max=2.5, initial=1.5 }
strength = { min=1.0, max=4.5, initial=1.0 }
growth_rate = { min=0.0, max=10.0, initial=1.0 }

[parameters.potential_type.Mie]
en = { min=0.2, max=25.0, initial=6.0, individual=false}
em = { min=0.2, max=25.0, initial=5.5}
bound = 8.0

[optimization.differential_evolution]
seed = 0
tol = 1e-3

[optimization.differential_evolution.polish]
method = 'HansWurst'
max_iter = 122

[others]
# progressbar = false
"
            .to_string();
            Ok((settings1, toml_string))
        })
    }

    #[test]
    fn test_parsing_toml() {
        let (settings1, toml_string) = generate_test_settings().unwrap();
        let settings = Settings::from_toml_string(&toml_string).unwrap();
        approx::assert_abs_diff_eq!(settings1, settings);
    }

    #[test]
    fn test_bound_generation() {
        Python::initialize();
        let (settings, _) = generate_test_settings().unwrap();

        for n_agents in 1..10 {
            let infos = Python::attach(|py| settings.generate_optimization_infos(py, n_agents));
            let lower = infos.bounds_lower;
            let upper = infos.bounds_upper;
            assert_eq!(lower.len(), n_agents + 5);
            assert_eq!(upper.len(), n_agents + 5);
        }
    }
}

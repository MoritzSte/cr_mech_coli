Changelog
#########

cr_mech_coli 0.10.0 `(13.03.2026) <_static/changelog/0.10.0.diff>`_
-----------------------------------------------------------------

- Improve documentation of Rust code
- Added image generation capabilities
    - New :ref:`crm_gen` script
    - New :ref:`crm_gen` module
- Remove secondary paper previously tracked in repository
- Add docstrings for scripts

cr_mech_coli 0.9.0 `(30.12.2025) <_static/changelog/0.9.0.diff>`_
-----------------------------------------------------------------

- Changes to :ref:`crm_divide`
    - Intermediately store calculated results when optimizing
    - Adjust plots and parameters
    - Rescale domain to mimick real parameters
    - Properly estimate growth rates
    - Plot split costs in profiles
- Changes to :ref:`crm_fit`
    - Now plots profiles with likelihood
- Improve mask generation: Automatically aim to adjust zoom values
- Add files containing paper and results

cr_mech_coli 0.8.0 `(22.09.2025) <_static/changelog/0.8.0.diff>`_
-----------------------------------------------------------------

- Improve results of `crm_amir <scripts/crm_amir>`_ script
- Allow to explicitly set growth rates for new cells
- Fix functionality of `crm_divide <scripts/crm_divide>`_ script

cr_mech_coli 0.7.1 `(19.09.2025 14:36) <_static/changelog/0.7.1.diff>`_
-----------------------------------------------------------------------

- Exclude some files in distribution release due to large file sizes

cr_mech_coli 0.7.0 `(19.09.2025 14:04) <_static/changelog/0.7.0.diff>`_
-----------------------------------------------------------------------

- Extend methods of :class:`CellContainer`
- Use rayon as default parallelizer
- Adjust to new `cellular_raza` version
- New script `crm_divide <scripts/crm_divide>`_ to fit model to dividing bacteria
- Calculate profiles more efficiently for `crm_fit <scripts/crm_fit>`_
- Improve type hints of Rust-based code

cr_mech_coli 0.6.0 `(19.06.2025) <_static/changelog/0.6.0.diff>`_
-----------------------------------------------------------------

- Modularize `crm_fit <scripts/crm_fit>`_ script
- Adjust to new `cellular_raza` version

cr_mech_coli 0.5.3 `(12.05.2025) <_static/changelog/0.5.3.diff>`_
-----------------------------------------------------------------

- Extend `crm_amir <scripts/crm_amir>`_ script

cr_mech_coli 0.5.2 `(09.05.2025) <_static/changelog/0.5.2.diff>`_
-----------------------------------------------------------------

- Update dependencies
- Add performance metrics in documentation
- Allow picking new growth rates from distribution
- New script `crm_amir <scripts/crm_amir>`_ to fit model to bent rod
- New script `crm_perf_plots <scripts/crm_perf_plots>`_ to plot performance of simulations
- Update plots of `crm_multilayer <scripts/crm_multilayer>`_ script

cr_mech_coli 0.5.1 `(01.05.2025) <_static/changelog/0.5.1.diff>`_
-----------------------------------------------------------------

- New script `crm_fit <scripts/crm_fit>`_ for fitting microscopig images
- New script `crm_multilayer <scripts/crm_multilayer>`_ for calculating multilayer behaviour of a
  colony.
- Use `BTreeMap` instead of `HashMap` in `CellContainer` to preserve order in CellContainer.
- New functions to (de)serialize and load a CellContainer from stored results
- Implement growth rate reduction depending on neighbors
- Rewrite main `run_simulation` function to take list of agents instead of only parameters.
- Add new data sources for `crm_fit <scripts/crm_fit>`_ script

cr_mech_coli 0.5.0 `(14.02.2025) <_static/changelog/0.5.0.diff>`_
-----------------------------------------------------------------

- Include `presentation <_static/presentation/index.html>`_ from Jeti Seminar 02.02.2025
- Generalize Interaction to use either Morse or Mie Potential
- First attempt at estimating parameters
  `ref~692b7899 <https://github.com/jonaspleyer/cr_mech_coli/commit/692b78993b4738fc041ae15fa073e55d6e990b59>`_
- Enhance scripts for fitting model to data
- Extend `extract_positions` function

cr_mech_coli 0.4.3 `(31.01.2025 19:42) <_static/changelog/0.4.3.diff>`_
-----------------------------------------------------------------------

- Release first version to pypi

cr_mech_coli 0.4.2 `(31.01.2025 19:08) <_static/changelog/0.4.2.diff>`_
-----------------------------------------------------------------------

- Get Github actions testing to work

cr_mech_coli 0.4.1 `(31.01.2025 11:17) <_static/changelog/0.4.1.diff>`_
-----------------------------------------------------------------------

- Short intermediate release to publish initial version to pypi
  See release ``0.5.0`` for more in-depth notes.

cr_mech_coli 0.4.0 `(19.11.2024) <_static/changelog/0.4.0.diff>`_
-----------------------------------------------------------------

- Added first iteration of preprint
- Improved position extraction algorithm

cr_mech_coli 0.3.0 `(31.10.2024) <_static/changelog/0.3.0.diff>`_
-----------------------------------------------------------------

- Added fitting methods
    - Extend documentation
    - Conversion between pixel and length-positions
- Introduction of the :class:`CellContainer` class which bundles information about a time series of
  cellular agents.
- Small improvements in Imaging
- Remove unused jupyter scripts

cr_mech_coli 0.2.0 `(23.10.2024) <_static/changelog/0.2.0.diff>`_
-----------------------------------------------------------------

cr_mech_coli 0.1.0 (14.10.2024)
-------------------------------

- Initial commit

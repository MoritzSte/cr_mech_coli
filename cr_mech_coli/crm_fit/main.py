import cv2 as cv
import cr_mech_coli as crm
from cr_mech_coli import crm_fit
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import time
from pathlib import Path
import multiprocessing as mp
import argparse
import scipy as sp
from tqdm import tqdm
import warnings
import shutil

from .plotting import plot_interaction_potential, plot_profile, plot_distribution


# Create folder to store output
def get_out_folder(iteration: int | None, potential_type) -> Path:
    base = Path(f"./out/crm_fit/{potential_type.to_short_string()}")
    if iteration is not None:
        out = base / f"{iteration:04}"
    else:
        folders = sorted(glob(str(base / "*")))
        if len(folders) > 0:
            n = int(folders[-1].split("/")[-1]) + 1
        else:
            n = 0
        out = base / f"{n:04}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def exponential_growth(t, grate, x0):
    return x0 * np.exp(grate * t)


def estimate_growth_rates(iterations, lengths, settings, out_path):
    times = (
        (np.array(iterations) - np.min(iterations))
        / (np.max(iterations) - np.min(iterations))
        * settings.constants.t_max
    )

    popts = []
    pcovs = []
    growth_rates = []
    growth_rates_err = []
    for i in range(len(lengths[0])):
        x0 = lengths[0][i]
        xf = lengths[-1][i]
        popt, pcov = sp.optimize.curve_fit(
            exponential_growth,
            times,
            [length[i] for length in lengths],
            p0=(np.log(xf / x0) / settings.constants.t_max, x0),
        )
        popts.append(popt)
        pcovs.append(pcov)
        growth_rates.append(popt[0])
        growth_rates_err.append(np.sqrt(pcov[0, 0]))

    growth_rates = np.array(growth_rates)
    growth_rates_err = np.array(growth_rates_err)

    fig, ax = plt.subplots(figsize=(8, 8))
    crm.plotting.configure_ax(ax)
    ax.plot(times, lengths, color=crm.plotting.COLOR5)
    for popt, pcov in zip(popts, pcovs):
        ax.plot(
            times,
            exponential_growth(times, *popt),
            color=crm.plotting.COLOR3,
        )
        ax.fill_between(
            times,
            exponential_growth(times, *(popt - np.array([pcov[0, 0] ** 0.5, 0]))),
            exponential_growth(times, *(popt + np.array([pcov[0, 0] ** 0.5, 0]))),
            color=crm.plotting.COLOR1,
            alpha=0.3,
        )

    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Rod Length [pix]")

    fig.savefig(out_path / "estimated-growth-rates.png")
    fig.savefig(out_path / "estimated-growth-rates.pdf")

    ax.cla()
    crm.plotting.configure_ax(ax)
    ax.hist(growth_rates)
    fig.savefig(out_path / "estimated-growth-rates-distribution.png")
    fig.savefig(out_path / "estimated-growth-rates-distribution.pdf")

    return growth_rates, growth_rates_err


def transform_input_mask(
    colors_data, mask_data, iteration, cell_container, return_colors=False
):
    z8 = np.uint8(0)
    cells_at_iter = cell_container.get_cells_at_iteration(iteration)
    color_mapping = {np.uint8(0): (z8, z8, z8)}

    for color_old, id in zip(colors_data, sorted(cells_at_iter.keys())):
        color_new = cell_container.get_color(id)
        color_mapping[np.uint8(color_old)] = color_new

    def mapping(ns):
        res = []
        for n in ns:
            if n in color_mapping:
                res.append(color_mapping[n])
            else:
                new_color = crm.counter_to_color(n)
                color_mapping[n] = new_color
                res.append(new_color)

        return np.array(res)

    res = np.apply_along_axis(mapping, -1, mask_data.astype(np.uint8))
    if return_colors:
        return res, color_mapping
    else:
        return res


def plot_optimization_progression(evals, out):
    fig, ax = plt.subplots(figsize=(8, 8))
    crm.plotting.configure_ax(ax)
    evals = np.sort(evals)[::-1]
    ax.plot(
        np.arange(len(evals)),
        evals,
        label="Cost Function",
        color=crm.plotting.COLOR3,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=1,
        frameon=False,
    )
    ax.set_xlabel("Iterations")
    ax.set_xscale("log")
    fig.savefig(out / "optimization-progression.png")
    fig.savefig(out / "optimization-progression.pdf")
    plt.close(fig)


def crm_fit_main():
    parser = argparse.ArgumentParser(
        description="Fits the Bacterial Rods model to a system of cells."
    )
    parser.add_argument(
        "-i",
        "--iteration",
        type=int,
        default=None,
        help="Use existing output folder instead of creating new one",
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=-1, help="Number of threads"
    )
    parser.add_argument(
        "data",
        help="Directory containing initial and final snapshots with masks.",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        help="Folder to store all output in. If left unspecified, the output folder will be\
            generated via OUTPUT_FOLDER='./out/crm_fit/POTENTIAL_TYPE/ITERATION/' where ITERATION\
            is the next number larger than any already existing one and POTENTIAL_TYPE is obtained\
            from the settings.toml file",
    )
    parser.add_argument(
        "--skip-profiles",
        default=False,
        help="Skips Plotting of profiles for parameters",
        action="store_true",
    )
    parser.add_argument(
        "--skip-masks",
        default=False,
        help="Skips Plotting of masks and microscopic images",
        action="store_true",
    )
    parser.add_argument(
        "--skip-param-space",
        default=False,
        help="Skips visualization of parameter space",
        action="store_true",
    )
    parser.add_argument(
        "--skip-distributions",
        default=False,
        help="Skips plotting of distributions",
        action="store_true",
    )
    parser.add_argument(
        "--fit-growth-rates",
        default=False,
        help="Estimate individual growth rates initially by fitting an exponential curve to rod\
            lengths.",
        action="store_true",
    )
    parser.add_argument(
        "--skip-interaction-potential",
        default=False,
        help="Plot the shape of the interaction potential",
        action="store_true",
    )
    parser.add_argument(
        "--profiles-samples",
        type=int,
        default=40,
        help="Number of samples per parameter",
    )
    parser.add_argument(
        "--profiles-maxiter",
        type=int,
        default=20,
        help="Number of optimization steps for each profile point",
    )
    parser.add_argument(
        "--profiles-method",
        type=str,
        default="Nelder-Mead",
        help="Method to be used to minimize cost function in profiles",
    )
    parser.add_argument(
        "--profiles-pre-global",
        default=False,
        action="store_true",
        help="Perform a pre-optimization with the differential_evolution algorithm before locally minimizing.",
    )
    parser.add_argument(
        "--profiles-post-local-skip",
        default=False,
        action="store_true",
        help="Perform a local minimization with a specified optimization routine",
    )
    parser.add_argument(
        "--profiles-pre-maxiter",
        type=int,
        default=20,
        help="Number of iterations for global pre-optimization",
    )
    pyargs = parser.parse_args()
    if pyargs.workers == -1:
        pyargs.workers = mp.cpu_count()

    if pyargs.data is None:
        dirs = sorted(glob("data/*"))
        if len(dirs) == 0:
            raise ValueError('Could not find any directory inside "./data/"')
        data_dir = Path(dirs[0])
    else:
        data_dir = Path(pyargs.data)
    files_images = sorted(glob(str(data_dir / "images/*")))
    files_masks = sorted(glob(str(data_dir / "masks/*.csv")))

    crm.plotting.set_mpl_rc_params()

    # Try to read config file
    filename = data_dir / "settings.toml"
    settings = crm_fit.Settings.from_toml(filename)
    potential_type = settings.parameters.potential_type

    out = get_out_folder(pyargs.iteration, potential_type)
    if pyargs.output_folder is not None:
        out = Path(pyargs.output_folder)
    else:
        # Copy settings toml to output folder
        shutil.copyfile(filename, out / "settings.toml")

    interval = time.time()

    imgs = [cv.imread(fi) for fi in files_images]

    masks = [np.loadtxt(fm, delimiter=",", converters=float) for fm in files_masks]

    print(f"{time.time() - interval:10.4f}s Loaded data")
    interval = time.time()

    n_vertices = settings.constants.n_vertices
    domain_size = settings.constants.domain_size

    iterations_all = []
    positions_all = []
    lengths_all = []
    colors_all = []
    for mask, filename in tqdm(
        zip(masks, files_masks), total=len(masks), desc="Extract positions"
    ):
        try:
            pos, length, _, colors = crm.extract_positions(
                mask, n_vertices, domain_size=domain_size
            )
            positions_all.append(pos)
            lengths_all.append(length)
            iterations_all.append(int(Path(filename).stem.split("-")[0]))
            colors_all.append(colors)
        except ValueError as e:
            print("Encountered Error during extraction of positions:")
            print(filename)
            print(e)
            print("Omitting this particular result.")

    n_agents = len(positions_all[0])
    initial_params = settings.generate_optimization_infos(n_agents).initial_values

    positions_all = np.array(positions_all, dtype=np.float32)
    iterations_all = np.array(iterations_all, dtype=np.uint64) - iterations_all[0]
    settings.constants.n_saves = max(iterations_all) - 1

    growth_rates, _ = estimate_growth_rates(iterations_all, lengths_all, settings, out)
    # Write estimated growth rates to csv file
    np.savetxt(str(out / "estimate_growth_rates.csv"), growth_rates)

    if pyargs.fit_growth_rates:
        gr_ind, gr_count = settings.parameters.set_growth_rate(
            list(growth_rates), n_agents
        )

    print(f"{time.time() - interval:10.4f}s Calculated initial values")
    interval = time.time()

    filename = "final_params.toml"
    if (out / filename).exists():
        optimization_result = crm_fit.OptimizationResult.load_from_file(out / filename)
        if pyargs.fit_growth_rates:
            # Adjust the parameters of the optimization_result
            optimization_result.params = [
                *optimization_result.params[:gr_ind],
                *optimization_result.params[gr_ind + gr_count :],
            ]
    else:
        optimization_result = crm_fit.run_optimizer(
            iterations_all, positions_all, settings, pyargs.workers
        )

        # Store information in file
        optimization_result.save_to_file(out / filename)
        print(f"{time.time() - interval:10.4f}s Finished Parameter Optimization")

    interval = time.time()

    plot_optimization_progression(optimization_result.evals, out)

    displacement_error = settings.constants.displacement_error

    # Plot Cost function against varying parameters
    if not pyargs.skip_profiles:
        warnings.filterwarnings(
            "ignore",
            message="Maximum number of function evaluations has been exceeded.",
        )
        for n in range(len(optimization_result.params)):
            fig_ax = None
            fig_ax = plot_profile(
                n,
                (positions_all, iterations_all, settings),
                optimization_result,
                out,
                pyargs.workers,
                displacement_error,
                pyargs,
                fig_ax,
            )
            fig, _ = fig_ax
            plt.close(fig)

        warnings.filterwarnings(
            "default",
            message="Maximum number of function evaluations has been exceeded.",
        )
        print(f"{time.time() - interval:10.4f} Plotted Profiles")
        interval = time.time()

    if not pyargs.skip_interaction_potential:
        plot_interaction_potential(
            settings, optimization_result, positions_all.shape[1], out
        )

    settings.others = crm_fit.Others("")
    cell_container = crm_fit.run_simulation(
        optimization_result.params,
        positions_all[0],
        settings,
    )
    print()
    print(f"{time.time() - interval:10.4f} Ran Simulation")
    interval = time.time()

    if cell_container is None:
        raise ValueError("Best fit does not produce valid result.")

    iterations_all = cell_container.get_all_iterations()
    agents_predicted = cell_container.get_cells_at_iteration(iterations_all[-1])

    if not pyargs.skip_masks:

        def plot_snapshot(pos, img, snapshot_dir, name, pred=None):
            if pred is not None:
                alpha_mask = np.all(pred == np.array([0, 0, 0]), axis=2)
                alpha_mask = np.repeat(np.expand_dims(alpha_mask, axis=2), 3, axis=2)
                img = alpha_mask * img + (1 - alpha_mask) * (0.5 * img + 0.5 * pred)
                img = img.astype(np.uint8)
            for p in pos:
                p = crm.convert_cell_pos_to_pixels(p, domain_size, img.shape[:2])
                img = cv.polylines(
                    np.array(img),
                    [np.round(p[:, ::-1]).astype(int)],
                    isClosed=False,
                    color=(250, 250, 250),
                    thickness=1,
                )
            odir = out / snapshot_dir
            odir.mkdir(parents=True, exist_ok=True)
            cv.imwrite(f"{odir}/{name}.png", img)

        def plot_position_diff(p_exact, p_fit, iteration, shape):
            width_to_height = shape[0] / shape[1]
            fig, ax = plt.subplots(figsize=(8, width_to_height * 8))
            for p1i, p2i in zip(p_exact, p_fit):
                crm.plotting.configure_ax(ax)
                x1 = p1i[:, 0]
                y1 = p1i[:, 1]
                x2 = p2i[:, 0]
                y2 = p2i[:, 1]
                ax.plot(x1, y1, color=crm.plotting.COLOR5, label="Exact")
                ax.plot(x2, y2, color=crm.plotting.COLOR3, label="Fit")

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles[:2],
                labels[:2],
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                frameon=False,
            )
            ax.set_xlim(0, float(domain_size[0]))
            ax.set_ylim(0, float(domain_size[1]))
            iterdir = out / "celldiffs"
            iterdir.mkdir(parents=True, exist_ok=True)
            fig.savefig(iterdir / f"cell-{iteration:06}.png")
            fig.savefig(iterdir / f"cell-{iteration:06}.pdf")
            plt.close(fig)

        def plot_mask_diff(
            colors_data, mask_data, iteration, cell_container: crm.CellContainer
        ):
            mask_transformed = transform_input_mask(
                colors_data, mask_data, iteration, cell_container
            )

            ppm = np.array(mask_transformed.shape[:2]) / np.array(domain_size)[::-1]
            rs = crm.RenderSettings(pixel_per_micron=ppm)
            mask_predicted = crm.render_mask(
                cell_container.get_cells_at_iteration(iteration),
                cell_container.cell_to_color,
                domain_size,
                rs,
            )

            mask_diff = crm.parents_diff_mask(
                mask_predicted,
                mask_transformed,
                cell_container.color_to_cell,
                cell_container.parent_map,
                0,
            ).astype(np.uint8)
            odir = out / "celldiffs"
            cv.imwrite(
                filename=str(odir / f"diff-{iteration:06}.png"), img=mask_diff * 255
            )
            return mask_predicted

        for colors_data, mask_data, pos_exact, iter, img in tqdm(
            zip(colors_all, masks, positions_all, iterations_all, imgs),
            total=len(colors_all),
            desc="Render masks",
        ):
            agents = cell_container.get_cells_at_iteration(iter)
            pos = np.array([c[0].pos for c in agents.values()])
            mask_predicted = plot_mask_diff(
                colors_data, mask_data, iter, cell_container
            )
            plot_snapshot(
                pos, img, "snapshots", f"predicted-{iter:06}", pred=mask_predicted
            )
            plot_snapshot(pos_exact, img, "snapshots", f"exact-{iter:06}")
            plot_position_diff(pos_exact, pos, iter, img.shape)

        print(f"{time.time() - interval:10.4f}s Rendered Masks")
        interval = time.time()

    if not pyargs.skip_distributions:
        odir = out / "distributions"
        odir.mkdir(parents=True, exist_ok=True)
        infos = settings.generate_optimization_infos(positions_all.shape[1])
        for n, name, values in settings.get_parameters_distributions(
            n_agents, optimization_result
        ):
            plot_distribution(n, name, values, odir, infos)

        print(f"{time.time() - interval:10.4f}s Plotted Distributions")

import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import multiprocessing as mp
import cv2 as cv

import cr_mech_coli as crm
from cr_mech_coli import crm_fit
from cr_mech_coli.crm_divide.optimize import (
    calculate_profiles,
    minimize_de,
    minimize_lhs,
)
from cr_mech_coli.crm_divide.predict import (
    objective_function,
    objective_function_return_all,
)

crm.plotting.set_mpl_rc_params()


def preprocessing(data_dir, n_masks=None):
    """
    Performs preprocessing steps such as extract positions, load configuration files and sets
    parameters accordingly.
    """
    if n_masks is None:
        files_images = sorted(glob(str(data_dir / "images/*")))
        files_masks = sorted(glob(str(data_dir / "masks/*.csv")))
    else:
        files_images = list(sorted(glob(str(data_dir / "images/*"))))[:n_masks]
        files_masks = list(sorted(glob(str(data_dir / "masks/*.csv"))))[:n_masks]
    masks = [
        np.loadtxt(fm, delimiter=",", dtype=np.uint8, converters=float)
        for fm in files_masks
    ]
    iterations_data = np.array([int(s[-10:-4]) for s in files_images])
    iterations_data = iterations_data - np.min(iterations_data)

    settings = crm_fit.Settings.from_toml(data_dir / "settings.toml")
    n_vertices = settings.constants.n_vertices

    positions_all = []
    lengths_all = []
    colors_all = []
    for mask, filename in tqdm(
        zip(masks, files_masks), total=len(masks), desc="Extract positions"
    ):
        try:
            pos, length, _, colors = crm.extract_positions(
                mask, n_vertices, domain_size=settings.constants.domain_size
            )
            positions_all.append(np.array(pos, dtype=np.float32))
            lengths_all.append(length)
            colors_all.append(colors)
        except ValueError as e:
            print("Encountered Error during extraction of positions:")
            print(filename)
            print(e)
            print("Omitting this particular result.")

    settings.constants.n_saves = max(iterations_data) - 1

    domain_height = settings.domain_height
    for n, p in enumerate(positions_all):
        positions_all[n] = np.append(
            p,
            domain_height / 2 + np.zeros((*p.shape[:2], 1)),
            axis=2,
        ).astype(np.float32)

    return masks, positions_all, settings, iterations_data


def plot_mask_adjustment(
    output_dir, masks_data, positions_all, settings, iterations_data
):
    """
    This function is mainly used for debugging.
    It plots snapshots of how a given image is adjusted such that segmentation masks from data and
    numerically generated results can be compared directly.
    To do this, the tree of the numerical simulation needs to be matched to the tree of the data.
    """
    radius = 0.4782565
    strength = 0.01
    # en = 6.0
    # em = 0.5
    potential_stiffness = 1.0
    # damping = 2.0
    growth_rates = [
        0.005995107,
        0.0068584173,
        0.0070885874,
        0.009034319,
        0.007861354,
        0.008311217,
    ]
    spring_length_thresholds = [0.8, 0.8, 0.9, 0.9]
    new_growth_rates = [0.001] * 8
    x0 = [
        radius,
        strength,
        potential_stiffness,
        # en,
        # em,
        # damping,
        *growth_rates,
        *spring_length_thresholds,
        *new_growth_rates,
    ]

    args = (
        positions_all,
        settings,
        masks_data,
        iterations_data,
        0.5,
    )

    (
        masks_adjusted,
        masks_predicted,
        color_to_cell,
        parent_map,
        _,
    ) = objective_function_return_all(x0, *args, show_progressbar=True)

    (output_dir / "mask_adjustments").mkdir(parents=True, exist_ok=True)
    for (mask_predicted, overlap), mask_adjusted, mask_data, mask_iter in tqdm(
        zip(
            masks_predicted,
            masks_adjusted,
            masks_data,
            iterations_data,
        ),
        total=len(masks_adjusted),
        desc="Plot Adjustments",
    ):
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        axs[0, 0].set_axis_off()
        axs[0, 0].set_title("Mask Data")
        axs[0, 1].set_axis_off()
        axs[0, 1].set_title("Mask Predicted")
        axs[1, 0].set_axis_off()
        axs[1, 0].set_title("Mask Adjusted")
        axs[1, 1].set_axis_off()
        axs[1, 1].set_title("Diff")

        diff = crm.parents_diff_mask(
            mask_predicted, mask_adjusted, color_to_cell, parent_map, 0.5
        )

        axs[0, 0].imshow(mask_data)

        def ident_to_text(ident):
            try:
                return f"D({ident[1]})"
            except:
                return f"I({ident[0]})"

        colors = list(sorted(np.unique(mask_data)))[1:]
        for n, v in enumerate(colors):
            x, y = np.where(mask_data == v)
            pos = np.mean([x, y], axis=1)
            axs[0, 0].text(
                pos[1], pos[0], n + 1, color="white", fontfamily="sans-serif", size=10
            )

        for k, v in color_to_cell.items():
            x, y = np.where(np.all(mask_predicted == k, axis=2))
            pos = np.mean([x, y], axis=1)
            axs[0, 1].text(
                pos[1],
                pos[0],
                ident_to_text(v),
                color="white",
                fontfamily="sans-serif",
                size=10,
            )

        axs[0, 1].imshow(mask_predicted)
        axs[1, 0].imshow(mask_adjusted)

        for k, v in color_to_cell.items():
            x, y = np.where(np.all(mask_adjusted == k, axis=2))
            pos = np.mean([x, y], axis=1)
            axs[1, 0].text(
                pos[1],
                pos[0],
                ident_to_text(v),
                color="white",
                fontfamily="sans-serif",
                size=10,
            )

        axs[1, 1].imshow(1 - diff, cmap="Grays")

        fig.savefig(output_dir / f"mask_adjustments/{mask_iter:06}.png")
        plt.close(fig)


def plot_time_evolution(
    masks_predicted,
    new_masks,
    color_to_cell,
    parent_map,
    iterations_simulation,
    iterations_data,
    settings,
    output_dir,
    final_parameters,
    args,
):
    fig, ax = plt.subplots(figsize=(8, 8))
    crm.plotting.configure_ax(ax)

    def plot_result(
        masks_predicted,
        color_to_cell,
        parent_map,
        configs=[
            (0, crm.plotting.COLOR1, "0.0"),
            (0.5, crm.plotting.COLOR2, "0.5"),
            (1.0, crm.plotting.COLOR3, "1.0"),
        ],
    ):
        for parent_penalty, color, name in configs:
            penalties = [
                crm.penalty_area_diff_account_parents(
                    new_mask,
                    masks_predicted[iter][0],
                    color_to_cell,
                    parent_map,
                    parent_penalty,
                )
                for iter, new_mask in zip(iterations_data, new_masks)
            ]
            ax.plot(
                np.array([iterations_simulation[i] for i in iterations_data])
                * settings.constants.dt,
                penalties,
                marker="x",
                color=color,
                label=f"p={name}",
            )

    ax.axvline(x=40.0, color="gray", linestyle="--", alpha=0.8)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))

    plot_result(
        masks_predicted,
        color_to_cell,
        parent_map,
        configs=[(1.0, crm.plotting.COLOR5, "1.0 Optimized")],
    )

    # Generate new results with increased growth rate to showcase difference
    # Modify growth rates such that they are 10% too large
    params_mod = np.array(final_parameters)
    params_mod[3:8] *= 1.2
    params_mod[12:] *= 1.2
    (_, masks_predicted_2, color_to_cell_2, parent_map_2, _) = (
        objective_function_return_all(params_mod, *args, show_progressbar=True)
    )
    plot_result(masks_predicted_2, color_to_cell_2, parent_map_2)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        ncol=2,
        frameon=False,
    )

    ax.set_ylabel("Cost Function")
    ax.set_xlabel("Time [h]")
    fig.savefig(output_dir / "time-evolution.pdf")
    fig.savefig(output_dir / "time-evolution.png")
    plt.close(fig)


def plot_profiles(
    parameters: np.ndarray,
    bounds,
    labels: list,
    final_costs: tuple[float, float, float],
    args,
    output_dir,
    pyargs,
):
    """
    Calculates or loads profiles and plots them split by different cost contributions.
    """
    n_samples = pyargs.profiles_samples
    # First try loading results
    try:
        samples = np.load(output_dir / "profile-samples.npy")
        costs = np.load(output_dir / "profile-costs.npy")

        assert samples.shape == (n_samples, len(parameters))
        assert costs.shape[:2] == samples.shape
    except:
        samples, costs = calculate_profiles(
            parameters,
            bounds,
            n_samples,
            args,
            pyargs,
        )
        np.save(output_dir / "profile-costs.npy", costs)
        np.save(output_dir / "profile-samples.npy", samples)

    costs = np.array(costs).reshape((n_samples, len(parameters), 3))
    assert n_samples == samples.shape[0]
    assert len(parameters) == samples.shape[1]

    for n, p in zip(range(len(parameters)), parameters):
        fig, ax = plt.subplots(figsize=(8, 8))
        crm.configure_ax(ax)

        x = samples[:, n]
        y1 = costs[:, n, 0]
        y2 = costs[:, n, 1]
        y3 = costs[:, n, 2]
        filt = costs[:, n, 0] != np.nan
        filt2 = costs[:, n, 0] <= 40_000
        filt *= filt2

        # Cost with overlap
        cwo = y2[filt]
        # Cost without overlap
        cwoup = y2[filt] - y3[filt]
        # Cost with overlap and parent penalty = 1
        cwopp1 = y1[filt] + y3[filt]

        x = np.array(x)[filt]

        # Add entry for final cost
        # Sort entries by value of the parameter

        cwo_fin = final_costs[1]
        cwoup_fin = final_costs[1] - final_costs[2]
        cwopp1_fin = final_costs[0] + final_costs[2]

        cwo = np.array([*cwo, cwo_fin])
        cwoup = np.array([*cwoup, cwoup_fin])
        cwopp1 = np.array([*cwopp1, cwopp1_fin])

        # Correctly sort new values
        x = np.array([*x, p])
        sorter = np.argsort(x)
        x = x[sorter]

        cwo = cwo[sorter]
        cwoup = cwoup[sorter]
        cwopp1 = cwopp1[sorter]

        ax.plot(
            x,
            cwo,
            c=crm.plotting.COLOR3,
            label="Metric",
        )
        ax.plot(
            x,
            cwoup,
            c=crm.plotting.COLOR3,
            linestyle="--",
            label="No Overlap",
        )
        ax.plot(
            x,
            cwopp1,
            c=crm.plotting.COLOR3,
            linestyle=":",
            label="No Parent Penalty",
        )
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=2,
            frameon=False,
        )

        ax.scatter([parameters[n]], [cwo_fin], c=crm.plotting.COLOR5, marker="x")
        ax.scatter([parameters[n]], [cwoup_fin], c=crm.plotting.COLOR5, marker="x")
        ax.scatter([parameters[n]], [cwopp1_fin], c=crm.plotting.COLOR5, marker="x")

        ax.set_xlabel(labels[n])
        ax.set_ylabel("Cost Contribution")
        ax.ticklabel_format(axis="y", style="sci")

        odir = output_dir / "profiles"
        odir.mkdir(parents=True, exist_ok=True)
        fig.savefig(odir / f"profile-{n:06}.png")
        plt.close(fig)


def plot_timings(
    parameters,
    positions_all,
    settings,
    masks_data,
    iterations_data,
    output_dir,
    n_samples: int = 3,
):
    """
    Calculates and plots timings of individual steps utilized within the cost function
    """
    times = []
    for _ in tqdm(range(n_samples), total=n_samples, desc="Measure Timings"):
        times.append(
            objective_function(
                parameters,
                positions_all,
                settings,
                masks_data,
                iterations_data,
                parent_penalty=0.5,
                return_timings=True,
            )
        )

    data = np.array(
        [[times[i][j][0] for j in range(len(times[0]))] for i in range(len(times))]
    )
    data = (data[:, 1:] - data[:, :-1]) / 1e9
    mean = np.mean(data, axis=0)
    ind = np.argsort(mean)[::-1]
    mean = mean[ind]
    dmean = np.std(data, axis=0)[ind]
    labels = np.array([t[1] for t in times[0][1:]])[ind]
    perc = mean / np.sum(mean)
    dperc = (
        (dmean / np.sum(mean)) ** 2 + (np.sum(dmean) * mean / np.sum(mean) ** 2) ** 2
    ) ** 0.5

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_ylim(0, np.max(mean) * 1.15)
    crm.configure_ax(ax)
    b = ax.bar(labels, mean, color=crm.plotting.COLOR3)
    ax.bar_label(
        b,
        [f"{100 * p:.2f}%\n±{100 * dp:.2f}%" for p, dp in zip(perc, dperc)],
        label_type="edge",
        color=crm.plotting.COLOR5,
        weight="bold",
    )
    ax.set_ylabel("Time [s]")
    fig.savefig(output_dir / "timings.pdf")
    fig.savefig(output_dir / "timings.png")


def callback(intermediate_result):
    fun = intermediate_result.fun
    global evals
    evals.append(float(fun))


def run_optimizer(
    params,
    bounds,
    output_dir,
    iteration,
    args,
    pyargs,
):
    """
    Wrapper method to call :function:`.minimize_lhs` or :function:`.minimize_de`.
    """
    global evals
    evals = []

    # Try loading data
    if iteration is not None:
        result = np.loadtxt(output_dir / "optimize_result.csv")
        evals = np.loadtxt(output_dir / "optimization_evals.csv")
        final_parameters = result[:-1]
        final_cost = result[-1]
    else:
        res = pyargs.func(params, bounds, args, callback, pyargs)
        final_parameters, final_cost = res

        np.savetxt(output_dir / "optimize_result.csv", [*final_parameters, final_cost])
        np.savetxt(output_dir / "optimization_evals.csv", evals)

    return final_parameters, final_cost, evals


def plot_snapshots(
    iterations_data,
    masks_predicted,
    masks_adjusted,
    output_dir,
    color_to_cell,
    parent_map,
):
    """
    Plots snapshots and differences of the optimized system to the provided data
    """
    (output_dir / "masks_predicted").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks_adjusted").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks_diff").mkdir(parents=True, exist_ok=True)
    for n, m in enumerate(masks_predicted):
        cv.imwrite(f"{output_dir}/masks_predicted/{n:06}.png", m[0])
    for n, m2 in zip(iterations_data, masks_adjusted):
        m1 = masks_predicted[n][0]
        cv.imwrite(f"{output_dir}/masks_adjusted/{n:06}.png", m2)
        diff = (
            crm.parents_diff_mask(m1, m2, color_to_cell, parent_map, 0.5) * 255
        ).astype(np.uint8)
        cv.imwrite(f"{output_dir}/masks_diff/{n:06}.png", diff)


def default_parameters() -> tuple[list[float], list[tuple[float, float]]]:
    """
    Returns:
        tuple[list[float], list[tuple[float, float]]]: Initial values and corresponding bounds for
        optimization
    """
    x0 = [
        # Radius
        4.101427290706450846e-01,
        # Strength
        1.519383356511266880e-02,
        # Potential Stiffness
        5.476115854564481467e00,
        # Growth Rates
        7.340484653494815104e-03,
        5.983537936575702987e-03,
        9.199234242500513303e-03,
        1.258385848317429903e-02,
        9.111606891670301356e-03,
        9.419760459145443132e-03,
        # Spring length thresholds
        6.022767130698081228e-01,
        6.559190757201818212e-01,
        5.943833407186450701e-01,
        5.677175696361352886e-01,
        # New growth rates
        3.230658553125494159e-03,
        1.258922850097723284e-02,
        1.794461648761763728e-02,
        4.761491437329616605e-03,
        5.241558679868058013e-03,
        1.161985425461459394e-03,
        1.853940041329739385e-02,
        5.169072925677239971e-04,
    ]
    bounds = [
        # Radius
        (0.003, 1.0),
        # Strength
        (0.0, 0.6),
        # Potential Stiffness
        (0.0, 15.0),
        # Growth rates
        (0.0000, 0.1),
        (0.0000, 0.1),
        (0.0000, 0.1),
        (0.0000, 0.1),
        (0.0000, 0.1),
        (0.0000, 0.1),
        # Spring length thresholds
        (0.2, 2.0),
        (0.2, 2.0),
        (0.2, 2.0),
        (0.2, 2.0),
        # new growth rates
        (0.0000, 0.1),
        (0.0000, 0.1),
        (0.0000, 0.1),
        (0.0000, 0.1),
        (0.0000, 0.1),
        (0.0000, 0.1),
        (0.0000, 0.1),
        (0.0000, 0.1),
    ]
    return x0, bounds


def plot_growth_rate_distribution(final_parameters, output_dir):
    """
    Plots the distribution of growth_rates obtained from the optimization routine
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    growth_rates = final_parameters[3:9]
    new_growth_rates = final_parameters[14:22]

    data = np.array([*growth_rates, *new_growth_rates])

    _, bins, _ = ax.hist(data, color=crm.plotting.COLOR3, alpha=0.5)
    ax.cla()
    crm.configure_ax(ax)
    y, _, _ = ax.hist(
        growth_rates, color=crm.plotting.COLOR3, bins=bins, label="Mother"
    )
    ax.hist(
        new_growth_rates,
        color=crm.plotting.COLOR3,
        bins=bins,
        bottom=y,
        alpha=0.5,
        label="Daughter",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=2,
        frameon=False,
    )

    ax.set_ylabel("Growth Rate")
    fig.savefig(output_dir / "growth-rate-distribution.pdf")
    fig.savefig(output_dir / "growth-rate-distribution.png")
    plt.close(fig)


def crm_divide_main():
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
        "--output-dir",
        type=str,
        default="out/crm_divide/",
        help="Directory where to store results",
    )
    parser.add_argument(
        "--skip-profiles",
        action="store_true",
        help="Skip plotting of profiles",
    )
    parser.add_argument(
        "--skip-time-evolution",
        action="store_true",
        help="Skip plotting of the time evolution of costs",
    )
    parser.add_argument(
        "--skip-snapshots",
        action="store_true",
        help="Skip plotting of snapshots and masks",
    )
    parser.add_argument(
        "--skip-timings",
        action="store_true",
        help="Skip plotting of the timings",
    )
    parser.add_argument(
        "--skip-mask-adjustment",
        action="store_true",
        help="Skip plotting of the adjusted masks",
    )
    parser.add_argument(
        "--only-mask-adjustment",
        action="store_true",
        help="Only plot adjusted masks",
    )
    parser.add_argument(
        "--skip-distribution",
        action="store_true",
        help="Skip plotting of the distribution of growth rates",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=-1,
        help="Number of threads to utilize",
    )

    parsers_optim = parser.add_subparsers(required=True)
    parser_de = parsers_optim.add_parser(
        "DE",
        help="Use the differential_evolution algorithm for optimization",
    )
    parser_de.add_argument("--maxiter", type=int, default=100)
    parser_de.add_argument("--popsize", type=int, default=15)
    parser_de.add_argument("--recombination", type=float, default=0.6)
    parser_de.add_argument("--tol", type=float, default=0.01)
    parser_de.add_argument("--mutation-upper", type=float, default=1.2)
    parser_de.add_argument("--mutation-lower", type=float, default=0.0)
    parser_de.add_argument("--skip-polish", action="store_true")
    parser_de.add_argument("--polish-maxiter", type=int, default=20)
    parser_de.set_defaults(func=minimize_de)

    parser_lhs = parsers_optim.add_parser(
        "LHS",
        help="Use the Latin-Hypercube Sampling with some local minimization for optimization",
    )
    parser_lhs.add_argument("--samples", type=int, default=1000)
    parser_lhs.add_argument("--local-maxiter", type=int, default=50)
    parser_lhs.add_argument("--local-method", type=str, default="Nelder-Mead")
    parser_lhs.add_argument("--polish-skip", action="store_true")
    parser_lhs.add_argument("--polish-maxiter", type=int, default=100)
    parser_lhs.add_argument("--polish-method", type=str, default="Nelder-Mead")
    parser_lhs.set_defaults(func=minimize_lhs)

    parser.add_argument("--profiles-maxiter", type=int, default=20)
    parser.add_argument("--profiles-samples", type=int, default=60)
    parser.add_argument(
        "--profiles-optim-method", type=str, default="differential_evolution"
    )
    parser.add_argument("--profiles-lhs-sample-size", type=int, default=50)
    parser.add_argument("--profiles-lhs-maxiter", type=int, default=10)
    parser.add_argument("--data-dir", type=Path, default=Path("data/crm_divide/0001/"))
    pyargs = parser.parse_args()

    if pyargs.workers <= 0:
        pyargs.workers = mp.cpu_count()

    iteration = pyargs.iteration
    if pyargs.iteration is None:
        existing = glob(f"{pyargs.output_dir}/*")
        if len(existing) == 0:
            iteration = 0
        else:
            iteration = max([int(Path(i).name) for i in existing]) + 1
    output_dir = Path(f"{pyargs.output_dir}/{iteration:06}")

    # Create the directory if we had to choose a new one
    if pyargs.iteration is None:
        output_dir.mkdir(parents=True)

    masks_data, positions_all, settings, iterations_data = preprocessing(
        pyargs.data_dir
    )

    if not pyargs.skip_mask_adjustment or pyargs.only_mask_adjustment:
        plot_mask_adjustment(
            output_dir, masks_data, positions_all, settings, iterations_data
        )
        if pyargs.only_mask_adjustment:
            exit()

    x0, bounds = default_parameters()
    parent_penalty = 0.5
    args = (
        positions_all,
        settings,
        masks_data,
        iterations_data,
        parent_penalty,
    )

    final_parameters, final_cost, evals = run_optimizer(
        x0,
        bounds,
        output_dir,
        pyargs.iteration,
        args,
        pyargs,
    )

    crm_fit.plot_optimization_progression(evals, output_dir)

    if (
        not pyargs.skip_snapshots
        or not pyargs.skip_time_evolution
        or not pyargs.skip_profiles
    ):
        (
            masks_adjusted,
            masks_predicted,
            color_to_cell,
            parent_map,
            container,
        ) = objective_function_return_all(
            final_parameters, *args, show_progressbar=True
        )

        if not pyargs.skip_snapshots:
            plot_snapshots(
                iterations_data,
                masks_predicted,
                masks_adjusted,
                output_dir,
                color_to_cell,
                parent_map,
            )

        if not pyargs.skip_time_evolution:
            plot_time_evolution(
                masks_predicted,
                masks_adjusted,
                color_to_cell,
                parent_map,
                container.get_all_iterations(),
                iterations_data,
                settings,
                output_dir,
                final_parameters,
                args,
            )

    if not pyargs.skip_profiles:
        labels = [
            "Radius",
            "Strength",
            "Potential Stiffness",
            "Growth Rate 0",
            "Growth Rate 1",
            "Growth Rate 2",
            "Growth Rate 3",
            "Growth Rate 4",
            "Growth Rate 5",
            "Division Length 0",
            "Division Length 1",
            "Division Length 2",
            "Division Length 3",
            "Growth Rate 0-0",
            "Growth Rate 0-1",
            "Growth Rate 1-0",
            "Growth Rate 1-1",
            "Growth Rate 2-0",
            "Growth Rate 2-1",
            "Growth Rate 3-0",
            "Growth Rate 3-1",
        ]
        final_costs = objective_function(
            final_parameters,
            positions_all,
            settings,
            masks_data,
            iterations_data,
            return_split_cost=True,
            print_costs=False,
            show_progressbar=True,
        )
        np.save(output_dir / "optimization-final-costs", final_costs)
        plot_profiles(
            final_parameters,
            bounds,
            labels,
            final_costs,
            args,
            output_dir,
            pyargs,
        )

    if not pyargs.skip_timings:
        plot_timings(
            final_parameters,
            positions_all,
            settings,
            masks_data,
            iterations_data,
            output_dir,
        )

    if not pyargs.skip_distribution:
        plot_growth_rate_distribution(
            final_parameters,
            output_dir,
        )

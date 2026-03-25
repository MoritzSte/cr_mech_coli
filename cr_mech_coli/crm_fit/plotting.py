import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
from tqdm.contrib.concurrent import process_map
import cr_mech_coli as crm
import scipy as sp

from cr_mech_coli.plotting import COLOR3, COLOR5

from .crm_fit_rs import Settings, OptimizationResult, predict_calculate_cost


def pred_flatten_wrapper(args):
    parameters, iterations, positions_all, settings = args
    return predict_calculate_cost(parameters, positions_all, iterations, settings)


def prediction_optimize_helper(
    params_opt, param_single, n_param, positions_all, iterations, settings
):
    params_all = [0] * (len(params_opt) + 1)
    params_all[:n_param] = params_opt[:n_param]
    params_all[n_param] = param_single
    params_all[n_param + 1 :] = params_opt[n_param:]

    return predict_calculate_cost(params_all, positions_all, iterations, settings)


def optimize_around_single_param(opt_args):
    (
        all_params,
        bounds_lower,
        bounds_upper,
        n,
        param_single,
        args,
        pyargs,
        constraints,
    ) = opt_args

    params_opt = list(all_params)
    b_low = list(bounds_lower)
    b_upp = list(bounds_upper)
    constr = list(constraints)

    del params_opt[n]
    del b_low[n]
    del b_upp[n]
    del constr[n]

    if n + 2 == len(bounds_lower):
        constr[-1] = 0

    constraints = sp.optimize.LinearConstraint(constr, lb=0)
    bounds = [(b_low[i], b_upp[i]) for i in range(len(b_low))]

    fun = np.nan
    success = False
    x0 = params_opt

    if pyargs.profiles_pre_global:
        res = sp.optimize.differential_evolution(
            prediction_optimize_helper,
            x0=params_opt,
            args=(param_single, n, *args),
            bounds=bounds,
            disp=False,
            maxiter=pyargs.profiles_pre_maxiter,
            polish=False,
            workers=1,
            popsize=5,
            mutation=(0.5, 1.5),
            constraints=constraints,
        )
        x0 = res.x
        fun = res.fun
        success = res.success

    if not pyargs.profiles_post_local_skip:
        res = sp.optimize.minimize(
            prediction_optimize_helper,
            x0=x0,
            args=(param_single, n, *args),
            bounds=bounds,
            method=pyargs.profiles_method,
            options={
                "disp": False,
                "maxiter": pyargs.profiles_maxiter,
            },
            constraints=constraints,
        )
        fun = res.fun
        success = res.success

    return fun, success


def fill_confidence_levels(x, y, ax, fill=True, thresholds=[0.68, 0.90, 0.95]):
    thresh_prev = 0
    for i, q in enumerate(thresholds):
        thresh = sp.stats.chi2.ppf(q, 1)
        color = crm.plotting.COLOR3 if i % 2 == 0 else crm.plotting.COLOR5
        filt = y <= thresh
        lower = np.max(np.array([y, np.repeat(thresh_prev, len(y))]), axis=0)
        if fill:
            ax.fill_between(
                x,
                lower,
                np.repeat(thresh, len(lower)),
                where=filt,
                interpolate=True,
                color=color,
                alpha=0.3,
            )
        thresh_prev = thresh

    return thresh_prev


def plot_profile_from_data(
    x,
    y,
    p_fixed,
    final_cost,
    ax,
    name,
    short,
    units,
    displacement_error,
    ls_color=crm.plotting.COLOR3,
    fill=True,
    label=None,
    linestyle="--",
    filter_thresh=16,
):
    # Extend x and y by values from final_params and final cost
    x = np.append(x, p_fixed)
    y = np.append(y, final_cost)
    sorter = np.argsort(x)
    x = x[sorter]
    y = y[sorter]

    ax.set_title(name)
    ax.set_ylabel(f"PL({short}) - L(θ)")
    ax.set_xlabel(f"{short} [{units}]")
    ax.scatter(
        p_fixed,
        0,
        marker="x",
        color=COLOR5,
        alpha=0.7,
        s=12**2,
    )

    y = (y - final_cost) / displacement_error**2

    # Filter for nan and infinity values
    filt1 = np.logical_and(~np.isnan(y), np.isfinite(y))
    # Also filter large values which are optimization artifacts i.e. outliers
    filt2 = y <= filter_thresh
    filt = filt1 * filt2
    filt[0] = True
    # filt[-1] = True
    x = x[filt]
    y = y[filt]

    thresh_prev = fill_confidence_levels(x, y, ax, fill)

    crm.plotting.configure_ax(ax)
    ax.plot(
        x,
        # (y - final_cost) / displacement_error**2,
        y,
        color=ls_color,  # crm.plotting.COLOR3,
        linestyle=linestyle,
        label=label,
    )

    upper = np.min([4 * thresh_prev, 1.05 * np.max([np.max(y), thresh_prev])])
    lower = -0.05 * upper
    ax.set_ylim(lower, upper)

    nxmin = np.min(np.where(y <= upper))
    nxmax = np.max(np.where(y <= upper))
    xmin = x[nxmin]
    xmax = x[nxmax]
    dx = xmax - xmin

    ax.set_xlim(xmin - 0.05 * dx, xmax + 0.05 * dx)

    return x, y


def calculate_profile(
    n, x, name, n_workers, optimization_result, infos, args, pyargs, constraints
):
    pool_args = [
        (
            optimization_result.params,
            infos.bounds_lower,
            infos.bounds_upper,
            n,
            p,
            args,
            pyargs,
            constraints,
        )
        for p in x
    ]
    y = process_map(
        optimize_around_single_param,
        pool_args,
        desc=f"Profile: {name}",
        max_workers=n_workers,
    )
    y_values = [yi[0] for yi in y]
    filt = [yi[1] for yi in y]
    return np.array(y_values), np.array(filt)


def plot_profile(
    n: int,
    args: tuple[np.ndarray, list[int], Settings],
    optimization_result: OptimizationResult,
    out: Path,
    n_workers,
    displacement_error: float,
    pyargs,
    fig_ax=None,
):
    (positions_all, iterations, settings) = args
    infos = settings.generate_optimization_infos(positions_all.shape[1])
    bound_lower = infos.bounds_lower[n]
    bound_upper = infos.bounds_upper[n]
    param_info = infos.parameter_infos[n]
    constraints = np.array(settings.get_constraints(positions_all.shape[1]))

    if fig_ax is None:
        fig_ax = plt.subplots(figsize=(8, 8))
        fig, ax = fig_ax
    else:
        fig, ax = fig_ax
        fig.clf()

    (name, units, short) = param_info

    odir = out / "profiles"
    odir.mkdir(parents=True, exist_ok=True)

    savename = name.strip().lower().replace(" ", "-")
    try:
        x = np.load(odir / f"profile-{savename}-params.npy")
        y = np.loadtxt(odir / f"profile-{savename}")
    except:
        x = np.linspace(bound_lower, bound_upper, pyargs.profiles_samples)
        y, filt = calculate_profile(
            n, x, name, n_workers, optimization_result, infos, args, pyargs, constraints
        )
        np.savetxt(odir / f"profile-{savename}", y)
        np.save(odir / f"profile-{savename}-params", x)
        np.save(odir / f"profile-{savename}-filter", filt)

    final_params = optimization_result.params
    final_cost = optimization_result.cost

    plot_profile_from_data(
        x,
        y,
        final_params[n],
        final_cost,
        ax,
        name,
        short,
        units,
        displacement_error,
    )

    plt.savefig(f"{odir / name.lower().replace(' ', '-')}.png")
    plt.savefig(f"{odir / name.lower().replace(' ', '-')}.pdf")
    return (fig, ax)


def plot_mie_potential(x, r, en, em, strength, bound, cutoff, fig_ax, ls):
    def sigma(r, n, m):
        return (m / n) ** (1 / (n - m)) * r

    def C(n, m):
        return n / (n - m) * (n / m) ** (n / (n - m))

    def mie(t, r, n, m, strength, bound, cutoff):
        v0 = (
            strength * C(n, m) * ((sigma(r, n, m) / t) ** n - (sigma(r, n, m) / t) ** m)
        )
        ds = np.abs(
            strength
            * C(n, m)
            / sigma(r, n, m)
            * (
                n * (sigma(1, n, m) / t) ** (n + 1)
                - m * (sigma(1, n, m) / t) ** (m + 1)
            )
        )
        co = t <= cutoff
        n = np.argmin(co)
        bo = ds <= bound
        m = np.argmax(bo)
        s = v0 * bo + (1 - bo) * (bound * (t[m] - t) + v0[m])
        return s * co + (1 - co) * v0[n], m

    if fig_ax is None:
        crm.plotting.set_mpl_rc_params()
        fig, ax = plt.subplots(figsize=(8, 8))
        crm.plotting.configure_ax(ax)
    else:
        fig, ax = fig_ax

    yfinmax = -np.inf
    y, m_bound = mie(x, r, en, em, strength, bound, cutoff)
    yfinmax = max(yfinmax, y[-1])
    ax.plot(x[: m_bound + 1], y[: m_bound + 1], linestyle=ls, color=crm.plotting.COLOR2)
    ax.plot(
        x[m_bound:],
        y[m_bound:],
        label=f"n={en:4.2f},m={em:4.2f}",
        linestyle=ls,
        color=crm.plotting.COLOR3,
    )
    ymin = 2 * np.max(np.abs(y))
    ax.vlines(cutoff, -ymin, yfinmax, color=crm.plotting.COLOR5)

    return fig, ax, y


def __plot_mie_potential(
    settings: Settings,
    optimization_result: OptimizationResult,
    n_agents,
    out,
    agent_index=0,
):
    en = settings.get_param("Exponent n", optimization_result, n_agents, agent_index)
    em = settings.get_param("Exponent m", optimization_result, n_agents, agent_index)
    r = 2 * settings.get_param("Radius", optimization_result, n_agents, agent_index)
    strength = settings.get_param(
        "Strength", optimization_result, n_agents, agent_index
    )
    bound = settings.get_param("Bound", optimization_result, n_agents, agent_index)
    cutoff = settings.constants.cutoff

    x = np.linspace(0.05 * r, 1.2 * settings.constants.cutoff, 500)
    fig, ax, y = plot_mie_potential(x, r, en, em, strength, bound, cutoff, None, "-")

    ymax = np.max(y)
    ymin = np.min(y)
    dy = ymax - ymin
    ax.set_ylim(ymin - 0.2 * dy, ymax + 0.2 * dy)

    # ax.plot(x / radius, y / strength, label="Mie Potential", color=crm.plotting.COLOR3)
    ax.set_xlabel("Distance [µm]")
    ax.set_ylabel("Interaction Strength [µm^2/min^2]")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=1,
        frameon=False,
    )

    fig.savefig(out / "potential-shape.png")
    fig.savefig(out / "potential-shape.pdf")
    plt.close(fig)


def plot_morse_potential(
    x,
    radius,
    potential_stiffness,
    strength,
    cutoff,
    fig_ax,
    ls,
    label=None,
    yoffset=0,
):
    def morse_potential(x, r, potential_stiffness, cutoff):
        t = 1 - np.exp(-potential_stiffness * (x - r))
        c = x <= cutoff
        n_last = np.argmax(1 - c)
        y = strength * t**2 + yoffset
        return y * c + (1 - c) * y[n_last], n_last

    if fig_ax is None:
        crm.plotting.set_mpl_rc_params()
        fig, ax = plt.subplots(figsize=(8, 8))
        crm.plotting.configure_ax(ax)
    else:
        fig, ax = fig_ax

    y, n_y_bound = morse_potential(x, radius, potential_stiffness, cutoff)
    ax.set_xlabel("Distance [µm]")
    ax.set_ylabel("Interaction Strength [µm^2/min^2]")

    # ax.plot(x, y, linestyle=ls, color=crm.plotting.COLOR2)
    if label is None:
        label = f"ω={potential_stiffness:4.2f}"
    ax.plot(
        x,
        y,
        label=label,
        linestyle=ls,
        color=crm.plotting.COLOR3,
    )

    ylower = np.min(y)
    yupper = np.max(y)
    dy = yupper - ylower
    ax.set_ylim(ylower - 0.05 * dy, yupper + 0.05 * dy)

    # n_y_bound = len(y) - np.argmax(y[::-1] > 0)
    yfinmax = y[n_y_bound]
    ax.vlines(cutoff, ylower - dy, yfinmax, color=crm.plotting.COLOR5)

    return fig, ax, y


def __plot_morse_potential(
    settings: Settings,
    optimization_result: OptimizationResult,
    n_agents,
    out,
    agent_index=0,
):
    r = settings.get_param("Radius", optimization_result, n_agents, agent_index)
    potential_stiffness = settings.get_param(
        "Potential Stiffness",
        optimization_result,
        n_agents,
        agent_index,
    )
    strength = settings.get_param(
        "Strength",
        optimization_result,
        n_agents,
        agent_index,
    )
    cutoff = settings.constants.cutoff

    x = np.linspace(0, 1.2 * settings.constants.cutoff, 500)

    crm.plotting.set_mpl_rc_params()
    fig, ax = plt.subplots(figsize=(8, 8))
    crm.plotting.configure_ax(ax)

    plot_morse_potential(
        x,
        r,
        potential_stiffness,
        strength,
        cutoff,
        (fig, ax),
        "-",
        label=None,
    )

    fig.savefig(out / "potential-shape.png")
    fig.savefig(out / "potential-shape.pdf")
    plt.close(fig)


def plot_interaction_potential(
    settings: Settings,
    optimization_result: OptimizationResult,
    n_agents,
    out,
):
    potential = settings.parameters.potential_type.to_short_string()
    if potential == "morse":
        __plot_morse_potential(settings, optimization_result, n_agents, out)
    if potential == "mie":
        __plot_mie_potential(settings, optimization_result, n_agents, out)


def plot_distribution(n, name, values, out, infos):
    fig, ax = plt.subplots(figsize=(8, 8))
    crm.configure_ax(ax)
    ax.hist(values, color=COLOR3, edgecolor=COLOR3, alpha=0.6)

    param_info = infos.parameter_infos[n]
    (_, units, short) = param_info

    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_title(name)
    ax.set_xlabel(f"{short} [{units}]")
    ax.set_ylabel("Count")

    xlower = np.min(values)
    xupper = np.max(values)
    dx = xupper - xlower

    ax.set_xlim(xlower - 0.05 * dx, xupper + 0.05 * dx)

    savename = name.lower().replace(" ", "-")

    fig.savefig(out / f"{savename}.png")
    fig.savefig(out / f"{savename}.pdf")
    fig.clf()

import numpy as np
import time
from tqdm import tqdm

import cr_mech_coli as crm
from cr_mech_coli import crm_fit
from cr_mech_coli import crm_divide


def adjust_masks(
    container: crm.CellContainer,
    masks_data: list,
    iterations_data: list,
    positions_all: list,
) -> tuple[list[np.ndarray], dict, dict]:
    mappings, color_to_cell, parent_map = crm_divide.get_color_mappings(
        container,
        masks_data,
        iterations_data,
        positions_all,
    )
    cell_to_color = {v: k for k, v in color_to_cell.items()}

    new_masks = []

    sim_iterations = container.get_all_iterations()
    for i, n in list(enumerate(iterations_data)):
        sim_iter = sim_iterations[n]
        mask_data = masks_data[i]

        mask_data_new = np.zeros((*mask_data.shape, 3), dtype=np.uint8)
        mapping = mappings[sim_iter]
        for color, cellident in mapping.items():
            new_color = cell_to_color[cellident]
            mask_data_new[mask_data == color] = new_color

        new_masks.append(mask_data_new)

    return new_masks, color_to_cell, parent_map


def predict(
    params,
    initial_positions,
    settings: crm_fit.Settings,
    show_progress=False,
) -> crm.CellContainer:
    (radius, strength, potential_stiffness) = params[:3]
    growth_rates = params[3:9]
    spring_length_thresholds = [*params[9:13], np.inf, np.inf]
    growth_rates_new = [
        *np.array(params[13:21]).reshape((-1, 2)),
        # These should not come into effect at all
        (0.0, 0.0),
        (0.0, 0.0),
    ]

    config = settings.to_config()
    if show_progress:
        config.progressbar = "Run Simulation"

    # Define agents
    interaction = crm.MorsePotentialF32(
        radius,
        potential_stiffness,
        settings.constants.cutoff,
        strength,
    )

    def spring_length(pos):
        dx = np.linalg.norm(pos[1:] - pos[:-1], axis=1)
        return np.mean(dx)

    agents = [
        crm.RodAgent(
            pos,
            vel=0 * pos,
            interaction=interaction,
            diffusion_constant=0.0,
            spring_tension=settings.parameters.spring_tension.get_inner(),
            rigidity=settings.parameters.rigidity.get_inner(),
            spring_length=spring_length(pos),
            damping=settings.parameters.damping.get_inner(),
            growth_rate=growth_rate,
            growth_rate_setter={"g1": g1, "g2": g2},
            spring_length_threshold=spring_length_threshold,
            spring_length_threshold_setter={"l1": np.inf, "l2": np.inf},
            neighbor_reduction=None,
        )
        for spring_length_threshold, pos, growth_rate, (g1, g2) in zip(
            spring_length_thresholds,
            initial_positions,
            growth_rates,
            growth_rates_new,
        )
    ]

    container = crm.run_simulation_with_agents(config, agents)
    if show_progress:
        print()

    return container


ERROR_COST = 1e7


def objective_function(
    params: list[float] | np.ndarray[tuple[int], np.dtype[np.float32]],
    positions_all: list[np.ndarray[tuple[int, int, int], np.dtype[np.float32]]],
    settings: crm_fit.Settings,
    masks_data: list[np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]],
    iterations_data: list[int],
    parent_penalty=0.5,
    return_all=False,
    return_timings=False,
    show_progressbar=False,
    print_costs=True,
    return_split_cost=False,
):
    times = [(time.perf_counter_ns(), "Start")]

    def update_time(message):
        if return_timings:
            now = time.perf_counter_ns()
            times.append((now, message))

    try:
        container = predict(
            params,
            positions_all[0],
            settings,
            show_progress=show_progressbar,
        )
    except ValueError or KeyError as e:
        if return_all:
            raise e
        if print_costs:
            print(f"f(x)={ERROR_COST:10.1f}")
        return ERROR_COST
    iterations_simulation = np.array(container.get_all_iterations()).astype(int)

    update_time("Predict")

    try:
        new_masks, color_to_cell, parent_map = adjust_masks(
            container,
            masks_data,
            iterations_data,
            positions_all,
        )
    except Exception:
        if print_costs:
            print(f"f(x)={ERROR_COST:10.1f}")
        return ERROR_COST

    update_time("Masks\n(Adjust)")

    pixel_per_micron = np.array(new_masks[0].shape[:2])[::-1] / np.array(
        settings.constants.domain_size
    )
    iters_filtered = np.array([iterations_simulation[i] for i in iterations_data])
    resolution = (
        int(pixel_per_micron[0] * settings.constants.domain_size[0]),
        int(pixel_per_micron[1] * settings.constants.domain_size[1]),
    )
    try:
        masks_predicted = [
            crm.render_mask_2d(
                container.get_cells_at_iteration(iter),
                {v: k for k, v in color_to_cell.items()},
                (settings.constants.domain_size[0], settings.constants.domain_size[1]),
                resolution,
                delta_angle=np.float32(np.pi / 8.0),
            )
            for iter in (iterations_simulation if return_all else iters_filtered)
        ]
    except ValueError:
        if print_costs:
            print(f"f(x)={ERROR_COST:10.1f}")
        return ERROR_COST

    update_time("Masks\n(Render)")

    # Remove overlaps from adjusted masks
    for i, m in enumerate(new_masks):
        overlap = masks_predicted[i][1]
        m[np.any(overlap != 0, axis=2)] = 0

    update_time("Remove Overlaps")

    # If we return all we need to filter the generated masks
    if return_all:
        mask_iterator = zip(
            [masks_predicted[iter] for iter in iterations_data], new_masks
        )
    # Otherwise we can use the whole list
    else:
        mask_iterator = zip(masks_predicted, new_masks)

    diff_masks = np.array(
        [
            crm.parents_diff_mask(
                m1[0],
                m2,
                color_to_cell,
                parent_map,
                parent_penalty,
            )
            for m1, m2 in mask_iterator
        ]
    )

    update_time("Calculate Diff Masks")

    overlaps = []
    for _, overlap_mask in (
        [masks_predicted[it] for it in iterations_data]
        if return_all
        else masks_predicted
    ):
        overlaps.append(np.sum(np.any(overlap_mask != 0, axis=2)))
    overlaps = np.array(overlaps)

    penalties = np.sum(diff_masks, axis=(1, 2)) + overlaps

    update_time("Compare")

    if return_all:
        return (
            new_masks,
            masks_predicted,
            color_to_cell,
            parent_map,
            container,
        )

    if return_timings:
        return times

    n_cells = len(container.get_cells_at_iteration(iterations_simulation[-1]))
    # cost = np.sum(penalties) * (1 + (n_cells - 10) ** 2) ** 0.5
    cost = np.sum(penalties)

    if print_costs:
        print(f"f(x)={cost:>10.1f}  Final Cells: {n_cells:2} Penalties: ", end="")
        for p in penalties:
            print(f" {p / np.sum(penalties) * 100:<4.2f}%", end="")
        print()

    if return_split_cost:
        costs_penalty_is_one = np.sum([np.sum(d != 0) for d in diff_masks])
        return costs_penalty_is_one, cost, np.sum(overlaps)
    else:
        return cost


def objective_function_return_all(
    *args, **kwargs
) -> tuple[list, list, dict, dict, crm.CellContainer]:
    return objective_function(*args, **kwargs, return_all=True)

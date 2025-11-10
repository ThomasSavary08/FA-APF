# Libraries
import dataclasses
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import xarray

from typing import List, Optional, Tuple, Union

from .wrapper.graphcast import checkpoint, data_utils, gencast


def custom_ticks(num_points: int):
    """
    Custom ticks for plots
    Input(s)
        - num_points (int): total number of points
    """
    tick_positions = np.arange(1, num_points, 2)
    tick_labels = np.arange(1, len(tick_positions) + 1)
    return tick_positions, tick_labels


def select(
    dataset: xarray.Dataset,
    variable: str,
    time_step: int,
    level: Optional[int] = None,
) -> xarray.DataArray:
    """
    Function to select data from an xarray (which may correspond to a complete trajectory)
    Input(s)
        - data (xarray.Dataset): dataset from which data is extracted
        - variable (str): variable to extract
        - time_step (int): indice(s) of the time steps to extract
        - level (int): level to extract (for atmospheric variables)
    """
    # Select the variable
    data = dataset[variable]

    # Remove the batch dimension if needed
    if "batch" in data.dims:
        data = data.isel(batch=0)

    # Select the time steps
    if "time" in data.sizes:
        data = data.isel(time=time_step)

    # Select the level
    if level is not None and "level" in data.coords:
        data = data.sel(level=level)

    # Convert longitude from [0,359] to [-180,180]
    if "lon" in data.coords:
        lon = data["lon"].values
        lon = ((lon + 180) % 360) - 180  # remap to [-180,180)
        data = data.assign_coords(lon=lon).sortby("lon")

    return data


def get_scale_and_cmap(
    data: xarray.DataArray,
    robust: bool = True,
) -> tuple[matplotlib.colors.Normalize, str]:
    """
    Function to get the scale and the cmap before plotting the data
    Input(s)
        - data (xarray.DataArray): data to display
        - robust (bool): if True, the 2 and 98 percentiles are taken to be robust to outliers
    """
    # Select v_min and v_max
    vmin = np.nanpercentile(data, (2 if robust else 0))
    vmax = np.nanpercentile(data, (98 if robust else 100))

    # Define the cmap
    cmap = "turbo"

    return matplotlib.colors.Normalize(vmin, vmax), cmap


def plot_trajectories(
    savepath: str,
    ground_truth: xarray.Dataset,
    ensemble_mean_filter: xarray.Dataset,
    ensemble_mean_unconditional: xarray.Dataset,
    time_steps: List[int],
    time_steps_titles: List[str],
    variable_to_plot: str,
    level: Optional[int] = None,
    robust: bool = True,
):
    """
    Plot ground truth (ERA5) and ensemble means (FA-APF and GenCast) for some variable and at some time steps.
    Input(s)
        - savepath (str): path to save the image
        - ground_truth (xarray.Dataset): reference from which observations are taken with dimension (batch=1, time=n, lat=181, lon=360, levels=13)
        - ensemble_mean_filter (xarray.Dataset): ensemble mean for the filter with dimension (batch=1, time=n, lat=181, lon=360, levels=13)
        - ensemble_mean_unconditional (xarray.Dataset): ensemble mean for GenCast with dimension (batch=1, time=n, lat=181, lon=360, levels=13)
        - time_steps (List[int]): time steps to plot with len(time_steps) <= 3
        - time_steps_titles (List[str]): title of the time steps
        - variable_to_plot (str): variable to plot
        - level (int): level if the variable to plot is not a surface variable
        - robust (bool): if True, the 2 and 98 percentiles are taken to be robust to outliers
    """
    # Check the length of time steps list
    assert len(time_steps) <= 3
    assert len(time_steps) == len(time_steps_titles)
    num_steps = len(time_steps) + 1
    time_steps = [0] + time_steps
    time_steps_titles = [r"$t_{0}$"] + time_steps_titles

    # Create the figure
    _, axs = plt.subplots(
        3, num_steps, figsize=(4 * num_steps, 6), squeeze=False, constrained_layout=True
    )
    row_labels = ["Ground Truth", "FA-APF", "GenCast"]

    # Loop on the steps
    for i, (step, step_title) in enumerate(zip(time_steps, time_steps_titles)):
        # Extract data from the ground_truth
        data_gt = select(
            dataset=ground_truth,
            variable=variable_to_plot,
            time_step=step,
            level=level,
        )

        # Extract data from the filter
        data_filter = select(
            dataset=ensemble_mean_filter,
            variable=variable_to_plot,
            time_step=step,
            level=level,
        )

        # Extract data from GenCast
        data_unconditional = select(
            dataset=ensemble_mean_unconditional,
            variable=variable_to_plot,
            time_step=step,
            level=level,
        )

        # Get the scale and cmap using the ground_truth
        norm, cmap = get_scale_and_cmap(data_gt, robust=robust)

        # Convert to numpy arrays (assume 2D lat x lon)
        arr_gt = np.array(data_gt)
        arr_filter = np.array(data_filter)
        arr_unconditional = np.array(data_unconditional)

        # Plot ground truth (first row)
        ax_gt = axs[0, i]
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        ax_gt.imshow(arr_gt, norm=norm, cmap=cmap, origin="lower")

        # Plot filter (second row)
        ax_fm = axs[1, i]
        ax_fm.set_xticks([])
        ax_fm.set_yticks([])
        ax_fm.imshow(arr_filter, norm=norm, cmap=cmap, origin="lower")

        # Plot unconditional (third row)
        ax_um = axs[2, i]
        ax_um.set_xticks([])
        ax_um.set_yticks([])
        ax_um.imshow(arr_unconditional, norm=norm, cmap=cmap, origin="lower")

        # Column title
        axs[0, i].set_title(step_title, fontsize=12)

    # Row labels
    for r in range(3):
        axs[r, 0].text(
            -0.08,
            0.5,
            row_labels[r],
            transform=axs[r, 0].transAxes,
            fontsize=12,
            va="center",
            rotation=90,
        )

    # Save the figure
    plt.savefig(savepath, bbox_inches="tight", dpi=150)
    plt.show()


def plot_surface_data(
    output_path: str,
    data_unconditional: xarray.Dataset,
    data_idealist: Union[xarray.Dataset, None],
    data_realistic: Union[xarray.Dataset, None],
    variables: List[str],
    num_row: int,
    num_col: int,
    title: str,
    figsize: Tuple[int],
    ylabels: List[str],
):
    """
    Function to plot data as a function of the lead time for different surface variables
    Input(s)
        - output_path (str): path to save the figure
        - data_unconditional (xarray.Dataset): data for unconditional trajectories
        - data_idealist (xarray.Dataset): data for idealist filtering (on a grid)
        - data_realistic (xarray.Dataset): data for reaalistic filtering (weather stations + satellites)
        - variables (List[str]): list of surface variables to plot
        - num_row (int): number of rows in the figure
        - num_col (int): number of columns in the figure
        - title (str): global title of the figure
        - figsize (Tuple[int]): size of the figure
        - ylabels (List[str]): labels to use for the figure
    """
    # Checks
    assert len(variables) == (num_row * num_col)
    if data_idealist is not None:
        assert int(data_unconditional.sizes["time"]) == int(data_idealist.sizes["time"])
    if data_realistic is not None:
        assert int(data_unconditional.sizes["time"]) == int(data_realistic.sizes["time"])
    assert len(variables) == len(ylabels)

    # Define plots
    plots = []
    for i, variable in enumerate(variables):
        # Get the data for the variable of interest
        data_unconditional_variable = data_unconditional[variable].data[0]
        if data_idealist is not None:
            data_idealist_variable = data_idealist[variable].data[0]
        else:
            data_idealist_variable = None
        if data_realistic is not None:
            data_realistic_variable = data_realistic[variable].data[0]
        else:
            data_realistic_variable = None

        # Define the plot
        plot = (
            data_unconditional_variable,
            data_idealist_variable,
            data_realistic_variable,
            ylabels[i],
        )
        plots.append(plot)

    # Plots
    fig, axs = plt.subplots(num_row, num_col, figsize=figsize, sharey=False)
    for ax, (data_unconditional, data_idealist, data_realistic, ylabel) in zip(axs, plots):
        # Custom ticks
        tick_positions, tick_labels = custom_ticks(len(data_unconditional))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

        # Plot the data
        ax.plot(data_unconditional, color="red", label="GenCast")
        if data_idealist is not None:
            ax.plot(data_idealist, color="blue", label="FA-APF (Idealist)")
        if data_realistic is not None:
            ax.plot(data_realistic, color="black", label="FA-APF (Realistic)")

        # Labels
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel("Lead time [days]", fontsize=12)
        ax.legend(fontsize=12)

    # Title
    fig.suptitle(title, fontsize=12, y=0.95)
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path)
    plt.show()


def plot_atmospheric_data(
    output_path: str,
    data_unconditional: Union[xarray.Dataset, None],
    data_idealist: Union[xarray.Dataset, None],
    data_realistic: Union[xarray.Dataset, None],
    variables: List[str],
    levels: List[int],
    title: str,
    figsize: Tuple[int],
    ylabels: List[str],
):
    """
    Function to plot data as a function of the lead time for different atmospheric variables
    Input(s)
        - output_path (str): path to save the figure
        - data_unconditional (xarray.Dataset): data for unconditional trajectories
        - data_idealist (xarray.Dataset): data for idealist filtering (on a grid)
        - data_realistic (xarray.Dataset): data for reaalistic filtering (weather stations + satellites)
        - variables (List[str]): list of atmospheric variables to plot
        - levels (List[int]): list of levels to plot
        - title (str): global title of the figure
        - figsize (Tuple[int]): size of the figure
        - ylabels (List[str]): labels to use for the figure
    """
    # Ckecks
    if (data_idealist is not None) and (data_unconditional is not None):
        assert int(data_unconditional.sizes["time"]) == int(data_idealist.sizes["time"])
    if (data_realistic is not None) and (data_unconditional is not None):
        assert int(data_unconditional.sizes["time"]) == int(data_realistic.sizes["time"])
    assert len(variables) == len(ylabels)

    fig, axs = plt.subplots(
        len(variables), len(levels), figsize=figsize, sharex=False, sharey=False
    )

    for i, variable in enumerate(variables):
        for j, level in enumerate(levels):
            ax = axs[i, j]

            # Get the data for the variable and level of interest
            if data_unconditional is not None:
                data_unconditional_variable = (
                    data_unconditional[variable].sel(level=int(level)).data[0]
                )
            else:
                data_unconditional_variable = None
            if data_idealist is not None:
                data_idealist_variable = data_idealist[variable].sel(level=int(level)).data[0]
            else:
                data_idealist_variable = None
            if data_realistic is not None:
                data_realistic_variable = data_realistic[variable].sel(level=int(level)).data[0]
            else:
                data_realistic_variable = None

            # Custom ticks
            if data_unconditional_variable is not None:
                tick_positions, tick_labels = custom_ticks(len(data_unconditional_variable))
            elif data_idealist_variable is not None:
                tick_positions, tick_labels = custom_ticks(len(data_idealist_variable))
            else:
                tick_positions, tick_labels = custom_ticks(len(data_realistic_variable))
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)

            # Plot the data
            if data_unconditional_variable is not None:
                ax.plot(data_unconditional_variable, color="red", label="GenCast")
            if data_idealist_variable is not None:
                ax.plot(data_idealist_variable, color="blue", label="FA-APF (Idealist)")
            if data_realistic_variable is not None:
                ax.plot(data_idealist_variable, color="black", label="FA-APF (Realistic)")

            # Labels of axis
            if j == 0:
                ax.set_ylabel(ylabels[i], fontsize=12)
            if i == len(variables) - 1:
                ax.set_xlabel("Lead time [days]", fontsize=12)
            if i == 0:
                ax.set_title(f"Level: {str(level)} hPa", fontsize=12)
            if i == 0 and j == 0:
                ax.legend(fontsize=12)

    # Title
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path)
    plt.show()


def get_metric(metric_path: str, num_steps: int) -> xarray.Dataset:
    """
    Load a metric computed at each time step for a filter
    Input(s)
        - metric_path (str): path to the metrics
        - num_steps (int): number of steps done during filtering
    Returns
        - ens_mean (xarray.Dataset): ensemble means with dimension (batch=1, time=num_steps, lat=181, lon=360, levels=13)
        - skill (xarray.Dataset): skill with dimension (batch=1, time=num_steps, levels=13)
        - spread (xarray.Dataset): spread with dimension (batch=1, time=num_steps, levels=13)
    """
    # Check the number of steps
    num_folders = sum(
        os.path.isdir(os.path.join(metric_path, name)) and not name.startswith(".")
        for name in os.listdir(metric_path)
    )
    assert num_folders == num_steps

    # Load the data
    ens_mean, skill, spread = [], [], []
    for step in range(1, num_steps + 1):
        # Define the common path
        if metric_path[-1] == "/":
            data_step_path = metric_path + str(step) + str("/")
        else:
            data_step_path = metric_path + str("/") + str(step) + str("/")

        # Define metrics path
        ens_step_path = data_step_path + str("ensemble_mean.nc")
        skill_step_path = data_step_path + str("skill.nc")
        spread_step_path = data_step_path + str("spread.nc")

        # Open the files
        with open(ens_step_path, "rb") as file:
            ens_mean_step = xarray.load_dataset(file, decode_timedelta=True).compute()
        with open(skill_step_path, "rb") as file:
            skill_step = xarray.load_dataset(file, decode_timedelta=True).compute()
        with open(spread_step_path, "rb") as file:
            spread_step = xarray.load_dataset(file, decode_timedelta=True).compute()

        # Update lists
        ens_mean.append(ens_mean_step)
        skill.append(skill_step)
        spread.append(spread_step)

    # Convert lists to xarray
    ens_mean = xarray.concat(ens_mean, dim="time")
    ens_mean = ens_mean.sortby("time")
    skill = xarray.concat(skill, dim="time")
    skill = skill.sortby("time")
    spread = xarray.concat(spread, dim="time")
    spread = spread.sortby("time")

    return ens_mean, skill, spread


def make_plots(
    num_steps: int,
    unconditional_path: str,
    idealist_path: Union[str, None],
    realistic_path: Union[str, None],
    gt_path: str,
    checkpoint_path: str,
    output_path: str,
    variables_first_plot: List[str],
    num_row_first_plot: int,
    num_col_first_plot: int,
    title_first_plot: str,
    figsize_first_plot: Tuple[int],
    ylabels_first_plot: List[str],
    variables_second_plot: List[str],
    levels_second_plot: List[str],
    title_second_plot: str,
    figsize_second_plot: Tuple[int],
    ylabels_second_plot: List[str],
    filter_third_plot: str,
    times_steps_third_plot: List[int],
    times_steps_titles_third_plot: List[str],
    variables_third_plot: List[Tuple],
    levels_third_plot: List,
):
    # Load the ground truth
    with open(gt_path, "rb") as file:
        data = xarray.load_dataset(file, decode_timedelta=True).compute()
    with open(checkpoint_path, "rb") as file:
        ckpt = checkpoint.load(file, gencast.CheckPoint)
    _, gt, _ = data_utils.extract_inputs_targets_forcings(
        data,
        target_lead_times=slice("12h", f"{(data.sizes['time'] - 2) * 12}h"),
        **dataclasses.asdict(ckpt.task_config),
    )

    # Load the unconditional data
    ens_unconditional, skill_unconditional, spread_unconditional = get_metric(
        metric_path=unconditional_path,
        num_steps=num_steps,
    )

    # Load idealist data
    if idealist_path is not None:
        ens_idealist, skill_idealist, spread_idealist = get_metric(
            metric_path=idealist_path,
            num_steps=num_steps,
        )
    else:
        ens_idealist, skill_idealist, spread_idealist = None, None, None

    # Load realistic data
    if realistic_path is not None:
        ens_realistic, skill_realistic, spread_realistic = get_metric(
            metric_path=realistic_path,
            num_steps=num_steps,
        )
    else:
        ens_realistic, skill_realistic, spread_realistic = None, None, None

    # First figure: skill for surface variables
    print("Plot skill for surface variables...")
    if output_path[-1] == "/":
        first_figure_path = output_path + str("fig1.svg")
    else:
        first_figure_path = output_path + str("/fig1.svg")
    plot_surface_data(
        output_path=first_figure_path,
        data_unconditional=skill_unconditional,
        data_idealist=skill_idealist,
        data_realistic=skill_realistic,
        variables=variables_first_plot,
        num_row=num_row_first_plot,
        num_col=num_col_first_plot,
        title=title_first_plot,
        figsize=figsize_first_plot,
        ylabels=ylabels_first_plot,
    )
    print("")

    # Second and third figures: spread and skill for atmospheric variables
    print("Plot skill and spread for atmospheric variables...")
    if output_path[-1] == "/":
        second_figure_path = output_path + str("fig2.svg")
        third_figure_path = output_path + str("fig3.svg")
    else:
        second_figure_path = output_path + str("/fig2.svg")
        third_figure_path = output_path + str("/fig3.svg")
    plot_atmospheric_data(
        output_path=second_figure_path,
        data_unconditional=skill_unconditional,
        data_idealist=skill_idealist,
        data_realistic=skill_realistic,
        variables=variables_second_plot,
        levels=levels_second_plot,
        title=title_second_plot,
        figsize=figsize_second_plot,
        ylabels=ylabels_second_plot,
    )
    plot_atmospheric_data(
        output_path=third_figure_path,
        data_unconditional=None,
        data_idealist=spread_idealist,
        data_realistic=spread_realistic,
        variables=variables_second_plot,
        levels=levels_second_plot,
        title=title_second_plot,
        figsize=figsize_second_plot,
        ylabels=ylabels_second_plot,
    )
    print("")

    # Last figures
    print("Plot ensemble means for GT, FA-APF and GenCast...")
    num_fig, times_steps_third_plot = 4, [int(elt) for elt in times_steps_third_plot]
    if filter_third_plot == "idealist":
        ensemble_mean_filter = ens_idealist
    else:
        ensemble_mean_filter = ens_realistic
    for i, variable in enumerate(variables_third_plot):
        if (levels_third_plot[i] is not None) and (levels_third_plot[i] != "None"):
            level = int(levels_third_plot[i])
        else:
            level = None
        if output_path[-1] == "/":
            figure_path = output_path + str("fig") + str(num_fig) + str(".svg")
        else:
            figure_path = output_path + str("/fig") + str(num_fig) + str(".svg")
        plot_trajectories(
            savepath=figure_path,
            ground_truth=gt,
            ensemble_mean_filter=ensemble_mean_filter,
            ensemble_mean_unconditional=ens_unconditional,
            time_steps=times_steps_third_plot,
            time_steps_titles=times_steps_titles_third_plot,
            variable_to_plot=variable,
            level=level,
            robust=True,
        )
        num_fig += 1

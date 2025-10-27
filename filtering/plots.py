# Libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray

from typing import List, Optional


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

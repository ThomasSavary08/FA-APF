# Libraries
import dataclasses
import matplotlib.pyplot as plt
import numpy as np
import xarray

from graphcast import checkpoint, data_utils, gencast
from plots import plot_trajectories


# Useful function for xlabel
def custom_ticks(n_points):
    tick_positions = np.arange(1, n_points, 2)
    tick_labels = np.arange(1, len(tick_positions) + 1)
    return tick_positions, tick_labels


# Parameters
num_steps = 30

# Load the reference
with open("./checkpoints/gencast_1deg.npz", "rb") as file:
    ckpt = checkpoint.load(file, gencast.CheckPoint)

with open("./data/trajectories/2019_03_29_1.0_13_30.nc", "rb") as file:
    example_batch = xarray.load_dataset(file, decode_timedelta=True).compute()

_, ground_truth, _ = data_utils.extract_inputs_targets_forcings(
    example_batch,
    target_lead_times=slice("12h", f"{(example_batch.sizes['time'] - 2) * 12}h"),
    **dataclasses.asdict(ckpt.task_config),
)


# Load filter skill
skill_filter_path = "./data/metrics/idealist/256_ERA5_2/"
skill_filter = []
for step in range(1, num_steps + 1):
    skill_step_path = skill_filter_path + str(step) + "/skill.nc"
    with open(skill_step_path, "rb") as file:
        skill_step = xarray.load_dataset(file, decode_timedelta=True).compute()
    skill_filter.append(skill_step)
skill_filter = xarray.concat(skill_filter, dim="time")
skill_filter = skill_filter.sortby("time")


# Load unconditional skill
skill_unconditional_path = "./data/metrics/unconditional/256/"
skill_unconditional = []
for step in range(1, num_steps + 1):
    skill_step_path = skill_unconditional_path + str(step) + "/skill.nc"
    with open(skill_step_path, "rb") as file:
        skill_step = xarray.load_dataset(file, decode_timedelta=True).compute()
    skill_unconditional.append(skill_step)
skill_unconditional = xarray.concat(skill_unconditional, dim="time")
skill_unconditional = skill_unconditional.sortby("time")

# Load filter spread
spread_filter_path = "./data/metrics/idealist/256_ERA5_2/"
spread_filter = []
for step in range(1, num_steps + 1):
    spread_step_path = spread_filter_path + str(step) + "/spread.nc"
    with open(spread_step_path, "rb") as file:
        spread_step = xarray.load_dataset(file, decode_timedelta=True).compute()
    spread_filter.append(spread_step)
spread_filter = xarray.concat(spread_filter, dim="time")
spread_filter = spread_filter.sortby("time")


# Load unconditional spread
spread_unconditional_path = "./data/metrics/unconditional/256/"
spread_unconditional = []
for step in range(1, num_steps + 1):
    spread_step_path = spread_unconditional_path + str(step) + "/spread.nc"
    with open(spread_step_path, "rb") as file:
        spread_step = xarray.load_dataset(file, decode_timedelta=True).compute()
    spread_unconditional.append(spread_step)
spread_unconditional = xarray.concat(spread_unconditional, dim="time")
spread_unconditional = spread_unconditional.sortby("time")

# Load filter ensemble_mean
mean_filter_path = "./data/metrics/idealist/256_ERA5_2/"
mean_filter = []
for step in range(1, num_steps + 1):
    mean_step_path = mean_filter_path + str(step) + "/ensemble_mean.nc"
    with open(mean_step_path, "rb") as file:
        mean_step = xarray.load_dataset(file, decode_timedelta=True).compute()
    mean_filter.append(mean_step)
mean_filter = xarray.concat(mean_filter, dim="time")
mean_filter = mean_filter.sortby("time")

#### A SUPPRIMER ####

# Load reference skill
skill_ref_path = "./data/metrics/idealist/256_ERA5/"
skill_ref = []
for step in range(1, num_steps + 1):
    skill_step_path = skill_ref_path + str(step) + "/skill.nc"
    with open(skill_step_path, "rb") as file:
        skill_step = xarray.load_dataset(file, decode_timedelta=True).compute()
    skill_ref.append(skill_step)
skill_ref = xarray.concat(skill_ref, dim="time")
skill_ref = skill_ref.sortby("time")

# Load reference spread
spread_ref = []
for step in range(1, num_steps + 1):
    spread_step_path = skill_ref_path + str(step) + "/spread.nc"
    with open(spread_step_path, "rb") as file:
        spread_step = xarray.load_dataset(file, decode_timedelta=True).compute()
    spread_ref.append(spread_step)
spread_ref = xarray.concat(spread_ref, dim="time")
spread_ref = spread_ref.sortby("time")

skill_t2m_ref = skill_ref["2m_temperature"].data[0]
skill_z500_ref = skill_ref["geopotential"].sel(level=[500]).data[0]
skill_10mu_ref = skill_ref["10m_u_component_of_wind"].data[0]

####


# Load unconditional ensemble_mean
mean_unconditional_path = "./data/metrics/unconditional/256/"
mean_unconditional = []
for step in range(1, num_steps + 1):
    mean_step_path = mean_unconditional_path + str(step) + "/ensemble_mean.nc"
    with open(mean_step_path, "rb") as file:
        mean_step = xarray.load_dataset(file, decode_timedelta=True).compute()
    mean_unconditional.append(mean_step)
mean_unconditional = xarray.concat(mean_unconditional, dim="time")
mean_unconditional = mean_unconditional.sortby("time")

# First plot (1,3): skills for specific variables (2m_temperature, z500, 10m_u_component_of_wind)
skill_t2m_filter = skill_filter["2m_temperature"].data[0]
skill_t2m_unconditional = skill_unconditional["2m_temperature"].data[0]

skill_z500_filter = skill_filter["geopotential"]
skill_z500_filter = skill_z500_filter.sel(level=[500]).data[0]
skill_z500_unconditional = skill_unconditional["geopotential"]
skill_z500_unconditional = skill_z500_unconditional.sel(level=[500]).data[0]

skill_10mu_filter = skill_filter["10m_u_component_of_wind"].data[0]
skill_10mu_unconditional = skill_unconditional["10m_u_component_of_wind"].data[0]

"""
plots = [
    (skill_t2m_filter, skill_t2m_unconditional, r"2m temperature $[\text{K}]$"),
    (skill_10mu_filter, skill_10mu_unconditional, r"10m U wind component $[\text{m}/\text{s}]$"),
    (
        skill_z500_filter,
        skill_z500_unconditional,
        r"Geopotential (500 hPa) $[\text{m}^{2}/\text{s}^{2}]$",
    ),
]

fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
for ax, (data_filter, data_uncond, ylabel) in zip(axs, plots):
    tick_positions, tick_labels = custom_ticks(len(data_uncond))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.plot(data_filter, color="blue", label="FA-APF")
    ax.plot(data_uncond, color="red", label="GenCast")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Lead time [days]", fontsize=12)
    ax.legend(fontsize=12)
fig.suptitle("Skill (Ensemble mean RMSE)", fontsize=12, y=0.95)
plt.tight_layout()
plt.savefig("./images/ERA5_2/fig1.png")
plt.show()
"""

plots = [
    (skill_t2m_filter, skill_t2m_unconditional, skill_t2m_ref, r"2m temperature $[\text{K}]$"),
    (
        skill_10mu_filter,
        skill_10mu_unconditional,
        skill_10mu_ref,
        r"10m U wind component $[\text{m}/\text{s}]$",
    ),
    (
        skill_z500_filter,
        skill_z500_unconditional,
        skill_z500_ref,
        r"Geopotential (500 hPa) $[\text{m}^{2}/\text{s}^{2}]$",
    ),
]

fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
for ax, (data_filter, data_uncond, data_ref, ylabel) in zip(axs, plots):
    tick_positions, tick_labels = custom_ticks(len(data_uncond))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.plot(data_filter, color="blue", label="FA-APF")
    ax.plot(data_uncond, color="red", label="GenCast")
    ax.plot(data_ref, color="green", label="Reference")  # NEW
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Lead time [days]", fontsize=12)
    ax.legend(fontsize=12)
fig.suptitle("Skill (Ensemble mean RMSE)", fontsize=12, y=0.95)
plt.tight_layout()
plt.savefig("./images/ERA5_2/fig1.png")
plt.show()

# Second plot (4,3) for appendices: skills for specific variables (T, Z, U, H) at three different levels (100, 250, 850)
variables = [
    ("temperature", "Temperature $[K]$"),
    ("geopotential", "Geopotential $[m^{2}/s^{2}]$"),
    ("v_component_of_wind", "V wind component $[m/s]$"),
    ("specific_humidity", "Specific humidity $[kg/kg]$"),
]
levels = [100, 250, 850]

fig, axs = plt.subplots(len(variables), len(levels), figsize=(15, 12), sharex=False, sharey=False)

for i, (var, ylabel) in enumerate(variables):
    for j, level in enumerate(levels):
        ax = axs[i, j]
        data_filter = skill_filter[var].sel(level=level).data[0]
        data_uncond = skill_unconditional[var].sel(level=level).data[0]
        tick_positions, tick_labels = custom_ticks(len(data_uncond))

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        data_ref = spread_ref[var].sel(level=level).data[0]
        ax.plot(data_ref, color="green", label="Reference")
        ax.plot(data_filter, color="blue", label="FA-APF")
        ax.plot(data_uncond, color="red", label="GenCast")

        if j == 0:
            ax.set_ylabel(ylabel, fontsize=12)
        if i == len(variables) - 1:
            ax.set_xlabel("Lead time [days]", fontsize=12)
        if i == 0:
            ax.set_title(f"Level: {level} hPa", fontsize=12)
        if i == 0 and j == 0:
            ax.legend(fontsize=12)

fig.suptitle(
    "Skill for temperature, geopotential, V wind component and humidity at three different pressure levels",
    fontsize=12,
)
plt.tight_layout()
plt.savefig("./images/ERA5_2/fig2.png")
plt.show()

# Third plot (4,3) for appendices: spread for specific variables (T, Z, U, H) at three different levels (100, 250, 850)
variables = [
    ("temperature", "Temperature $[K]$"),
    ("geopotential", "Geopotential $[m^{2}/s^{2}]$"),
    ("v_component_of_wind", "V wind component $[m/s]$"),
    ("specific_humidity", "Specific humidity $[kg/kg]$"),
]
levels = [100, 250, 850]

fig, axs = plt.subplots(len(variables), len(levels), figsize=(15, 12), sharex=False, sharey=False)

for i, (var, ylabel) in enumerate(variables):
    for j, level in enumerate(levels):
        ax = axs[i, j]
        data_filter = spread_filter[var].sel(level=level).data[0]
        tick_positions, tick_labels = custom_ticks(len(data_uncond))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.plot(data_filter, color="blue", label="FA-APF")

        if j == 0:
            ax.set_ylabel(ylabel, fontsize=12)
        if i == len(variables) - 1:
            ax.set_xlabel("Lead time [days]", fontsize=12)
        if i == 0:
            ax.set_title(f"Level: {level} hPa", fontsize=12)
        if i == 0 and j == 0:
            ax.legend(fontsize=12)

fig.suptitle(
    "Spread for temperature, geopotential, V wind component and humidity at three different pressure levels",
    fontsize=12,
)
plt.tight_layout()
plt.savefig("./images/ERA5_2/fig3.png")
plt.show()

# Final plots (3,4): ensemble_mean trajectories comparison for different variables
plot_trajectories(
    savepath="./images/ERA5_2/fig4.png",
    ground_truth=ground_truth,
    ensemble_mean_filter=mean_filter,
    ensemble_mean_unconditional=mean_unconditional,
    time_steps=[5, 13, 29],
    time_steps_titles=["+3 days", "+7 days", "+15 days"],
    variable_to_plot="2m_temperature",
)

plot_trajectories(
    savepath="./images/ERA5_2/fig5.png",
    ground_truth=ground_truth,
    ensemble_mean_filter=mean_filter,
    ensemble_mean_unconditional=mean_unconditional,
    time_steps=[5, 13, 29],
    time_steps_titles=["+3 days", "+7 days", "+15 days"],
    variable_to_plot="10m_u_component_of_wind",
)

plot_trajectories(
    savepath="./images/ERA5_2/fig6.png",
    ground_truth=ground_truth,
    ensemble_mean_filter=mean_filter,
    ensemble_mean_unconditional=mean_unconditional,
    time_steps=[5, 13, 29],
    time_steps_titles=["+3 days", "+7 days", "+15 days"],
    variable_to_plot="geopotential",
    level=500,
)

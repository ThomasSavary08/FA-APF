# Libraries
import dataclasses
import numpy as np
import tqdm
import xarray

from graphcast import checkpoint, data_utils, gencast

# Define paths
filter_path = "./data/filtering/idealist/256_ERA5/"
metrics_path = "./data/metrics/idealist/256_ERA5/"

# Load the reference
with open("./checkpoints/gencast_1deg.npz", "rb") as file:
    ckpt = checkpoint.load(file, gencast.CheckPoint)

with open("./data/trajectories/2019_03_29_1.0_13_30.nc", "rb") as file:
    example_batch = xarray.load_dataset(file, decode_timedelta=True).compute()

_, reference_ERA5, _ = data_utils.extract_inputs_targets_forcings(
    example_batch,
    target_lead_times=slice("12h", f"{(example_batch.sizes['time'] - 2) * 12}h"),
    **dataclasses.asdict(ckpt.task_config),
)

# Filter parameters
num_steps = 30
num_particles = 256

# 1) Compute the ensemble mean for each step
for step in range(1, num_steps + 1):
    particles = []
    step_path = filter_path + str(step) + "/"
    for i in tqdm.tqdm(range(1, num_particles + 1)):
        particle_path = step_path + str(i) + ".nc"
        with open(particle_path, "rb") as file:
            particle = xarray.load_dataset(file, decode_timedelta=True).compute()
        particle = particle.isel(time=[-1])
        particles.append(particle)
    particles = xarray.concat(particles, dim="batch")
    ensemble_mean = particles.mean(dim="batch", keepdims=True)
    file_path = metrics_path + str(step) + str("/ensemble_mean.nc")
    ensemble_mean.to_netcdf(file_path)

# 2) Compute the skill (RMSE of the ensemble mean)
for step in range(1, num_steps + 1):
    ensemble_mean_path = metrics_path + str(step) + "/ensemble_mean.nc"
    with open(ensemble_mean_path, "rb") as file:
        ensemble_mean = xarray.load_dataset(file, decode_timedelta=True).compute()
    reference_step = reference_ERA5.isel(time=[step - 1])
    skill = (reference_step - ensemble_mean) ** 2
    skill = skill.mean(dim=["lat", "lon"])
    skill = skill.map(np.sqrt)
    file_path = metrics_path + str(step) + str("/skill.nc")
    skill.to_netcdf(file_path)

# 3) Compute the Spread
for step in range(1, num_steps + 1):
    particles = []
    ensemble_mean_path = metrics_path + str(step) + "/ensemble_mean.nc"
    with open(ensemble_mean_path, "rb") as file:
        ensemble_mean = xarray.load_dataset(file, decode_timedelta=True).compute()
    step_path = filter_path + str(step) + "/"
    for i in tqdm.tqdm(range(1, num_particles + 1)):
        particle_path = step_path + str(i) + ".nc"
        with open(particle_path, "rb") as file:
            particle = xarray.load_dataset(file, decode_timedelta=True).compute()
        particle = particle.isel(time=[-1])
        particle = (particle - ensemble_mean) ** 2
        particles.append(particle)
    particles = xarray.concat(particles, dim="batch")
    spread = (1.0 / (num_particles - 1)) * particles.sum(dim="batch", keepdims=True)
    spread = spread.mean(dim=["lat", "lon"])
    spread = spread.map(np.sqrt)
    file_path = metrics_path + str(step) + str("/spread.nc")
    spread.to_netcdf(file_path)

# Libraries
import dataclasses
import fa_apf
import jax.numpy as jnp  # type: ignore
import os
import xarray

from graphcast import checkpoint, data_utils, gencast, samplers_utils

# Modify flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

################################################################
######################## Observations ##########################
################################################################
observed_variables = ["2m_temperature", "temperature"]
lat, lon = 181, 360
mask_rows = jnp.arange(lat) % 4 == 0
mask_cols = jnp.arange(lon) % 4 == 0
mask = jnp.outer(mask_rows, mask_cols)
sigma_y = jnp.full((14,), 0.1**2)


################################################################
##################### MMPS configuration #######################
################################################################
solver = "bicgstab"
max_iter_solver = 2
tol_solver = 1e-8


################################################################
############ checkpoint, x0, template and forcings #############
################################################################
with open("./checkpoints/gencast_1deg.npz", "rb") as file:
    ckpt = checkpoint.load(file, gencast.CheckPoint)

with open("./data/trajectories/2019_03_29_1.0_13_30.nc", "rb") as file:
    example_batch = xarray.load_dataset(file, decode_timedelta=True).compute()

x0, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch,
    target_lead_times=slice("12h", f"{(example_batch.sizes['time'] - 2) * 12}h"),
    **dataclasses.asdict(ckpt.task_config),
)


################################################################
######################## Architecture  #########################
################################################################
task_config = ckpt.task_config
noise_encoder_config = ckpt.noise_encoder_config
denoiser_architecture_config = ckpt.denoiser_architecture_config
denoiser_architecture_config.sparse_transformer_config.mask_type = "full"
denoiser_architecture_config.sparse_transformer_config.attention_type = "triblockdiag_mha"


################################################################
####################### Sampler config  ########################
################################################################
sampler = "abs"
noise_levels = samplers_utils.noise_schedule(
    max_noise_level=88.0,
    min_noise_level=3e-2,
    num_noise_levels=32,
    rho=6.0,  # 5
)
sampler_config = {
    "noise_levels": noise_levels,
    "order": 3,
    "correction": True,
    "num_correction_steps": 2,
    "delta": 0.2,
}


################################################################
######################### Statistics ###########################
################################################################
with open("./data/stats/std_x.nc", "rb") as file:
    std_x = xarray.load_dataset(file, decode_timedelta=True).compute()
with open("./data/stats/std_z.nc", "rb") as file:
    std_z = xarray.load_dataset(file, decode_timedelta=True).compute()
with open("./data/stats/mean_x.nc", "rb") as file:
    mean_x = xarray.load_dataset(file, decode_timedelta=True).compute()
with open("./data/stats/min_x.nc", "rb") as file:
    min_x = xarray.load_dataset(file, decode_timedelta=True).compute()


################################################################
######################## Filter design #########################
################################################################
filter_path = "./data/filtering/idealist/256_ERA5_2"
N = 256
N_thr_min = 70  # 60
N_thr_max = 80  # 70
alpha_init = 1.0 / (sigma_y.shape[0] * jnp.sum(mask))
max_iter_alpha = 100


################################################################
######################## Run filtering #########################
################################################################
fa_apf.filtering(
    filter_path=filter_path,
    N=N,
    N_thr_min=N_thr_min,
    N_thr_max=N_thr_max,
    alpha_init=alpha_init,
    reference=eval_targets,
    mask=mask,
    observed_variables=observed_variables,
    sigma_y=sigma_y,
    x0=x0,
    forcings=eval_forcings,
    target_template=eval_targets,
    ckpt=ckpt,
    task_config=task_config,
    denoiser_config=denoiser_architecture_config,
    noise_encoder_config=noise_encoder_config,
    sampler=sampler,
    sampler_config=sampler_config,
    std_z=std_z,
    min_x=min_x,
    std_x=std_x,
    mean_x=mean_x,
    noise_levels=noise_levels,
    max_iter_alpha=max_iter_alpha,
    solver=solver,
    max_iter_solver=max_iter_solver,
    tol_solver=tol_solver,
)

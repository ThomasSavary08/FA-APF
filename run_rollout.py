# Libraries
import dataclasses
import xarray

import parallel_rollout

from graphcast import checkpoint, data_utils, gencast

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
sampler = "dpm"
sampler_config = ckpt.sampler_config


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
######################### Run rollout ##########################
################################################################
num_trajectories = 256
traj_path = "./data/trajectories/unconditional"
parallel_rollout.generate_trajectories(
    N=num_trajectories,
    traj_path=traj_path,
    x0=x0,
    target_template=eval_targets,
    forcings=eval_forcings,
    ckpt=ckpt,
    task_config=task_config,
    denoiser_config=denoiser_architecture_config,
    noise_encoder_config=noise_encoder_config,
    sampler=sampler,
    sampler_config=sampler_config,
    min_x=min_x,
    std_x=std_x,
    std_z=std_z,
    mean_x=mean_x,
)

# Libraries
import dataclasses
import gc
import haiku as hk
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import numpy as np
import os
import tqdm
import warnings
import xarray

from jax import Array  # type: ignore
from typing import Dict, List, Union

import sampler as Samplers
import utils

from denoisers import ConditionalDenoiser
from graphcast import (
    checkpoint,
    data_utils,
    denoiser,
    gencast,
    graphcast,
    samplers_utils,
    xarray_jax,  # noqa: F401
)
from predictor import Predictor

# Modify flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# Ignore warnings of GenCast (about sparsity)
warnings.filterwarnings("ignore")

# Load the checkpoint
with open("./checkpoints/gencast_1deg.npz", "rb") as file:
    ckpt = checkpoint.load(file, gencast.CheckPoint)

# Load statistics
with open("./data/stats/std_x.nc", "rb") as file:
    std_x = xarray.load_dataset(file, decode_timedelta=True).compute()
with open("./data/stats/std_z.nc", "rb") as file:
    std_z = xarray.load_dataset(file, decode_timedelta=True).compute()
with open("./data/stats/mean_x.nc", "rb") as file:
    mean_x = xarray.load_dataset(file, decode_timedelta=True).compute()
with open("./data/stats/min_x.nc", "rb") as file:
    min_x = xarray.load_dataset(file, decode_timedelta=True).compute()

# Inputs, targets and forcings
with open("./data/trajectories/2019_03_29_1.0_13_12.nc", "rb") as file:
    example_batch = xarray.load_dataset(file, decode_timedelta=True).compute()

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch,
    target_lead_times=slice("12h", f"{(example_batch.sizes['time'] - 2) * 12}h"),
    **dataclasses.asdict(ckpt.task_config),
)

eval_targets = eval_targets.isel(time=[0])
eval_forcings = eval_forcings.isel(time=[0])

# Load the reference
with open("./data/trajectories/reference.nc", "rb") as file:
    reference = xarray.load_dataset(file, decode_timedelta=True).compute()
reference = reference.isel(time=[0])

# Sampler configuration
sampler_config = {
    "noise_levels": samplers_utils.noise_schedule(
        max_noise_level=88.0,
        min_noise_level=3e-4,
        num_noise_levels=60,
        rho=5.0,
    ),
    "order": 3,
    "correction": True,
    "num_correction_steps": 2,
    "delta": 0.25,
}

# Denoiser configuration
denoiser_architecture_config = ckpt.denoiser_architecture_config
denoiser_architecture_config.sparse_transformer_config.mask_type = "full"
denoiser_architecture_config.sparse_transformer_config.attention_type = "triblockdiag_mha"

# Observation configuration
mask = jnp.array(np.load("./data/observations/mask_1.npy").astype(bool))
observed_variables = ["2m_temperature"]
sigma_y = jnp.array([0.1]) ** 2

# MMPS configuration
solver = "bicgstab"
max_iter = 2
tol = 1e-8


# Function to draw samples from p(x_{k} | x_{k-1}^{(i)}, y_{k})
@hk.transform_with_state
def conditional_sampling(
    inputs: xarray.Dataset,
    target_template: xarray.Dataset,
    forcings: xarray.Dataset,
    task_config: graphcast.TaskConfig,
    denoiser_config: denoiser.DenoiserArchitectureConfig,
    noise_encoder_config: denoiser.NoiseEncoderConfig,
    sampler: str,
    sampler_config: Union[Dict, gencast.SamplerConfig],
    min_x: xarray.Dataset,
    std_x: xarray.Dataset,
    std_z: xarray.Dataset,
    mean_x: xarray.Dataset,
    reference: Array = None,
    mask: Array = None,
    observed_variables: List[str] = None,
    sigma_y: Array = None,
    solver: str = None,
    max_iter: int = None,
    tol: float = None,
) -> xarray.Dataset:
    """
    Draw a sample conditionally on an observation
    Input(s)
        - inputs (xarray.Dataset): previous states hat{x}_{t-2, t-1}^{(i)} of the system with dimensions (batch=1, time=2, lat=181, lon=360, levels=13)
        - target_template (xarray.Dataset): template with dimensions (batch=1, time=1, lat=181, lon=360, levels=13)
        - forcings (xarray.Dataset): forcings terms used by the GenCast denoiser with dimensions (batch=1, time=1, lat=181, lon=360, levels=13)
        - task_config (graphcast.TaskConfig)
        - denoiser_config (denoiser.DenoiserArchitectureConfig)
        - noise_encoder_config (denoiser.NoiseEncoderConfig)
        - sampler (str): sampler to use
        - sampler_config (Union[Any, gencast.SamplerConfig])
        - min_x (xarray.Dataset): minimum values of system states for each variable
        - std_x (xarray.Dataset): standard deviation of system states for each variable
        - std_z (xarray.Dataset): standard deviation of residuals for each variable
        - mean_x (xarray.Dataset): mean of system states for each variable
        - reference (xarray.Dataset): reference from which observations are extracted with dimensions (batch=1, time=1, lat=181, lon=360, levels=13)
        - mask (Array): mask used to do subsampling with dimension (181, 360)
        - observed_variables (List[str]): ordered list of observed variables
        - sigma_y (Array): covariance matrix of observations Sigma_{y} with dimension (len(observed_variables),)
        - solver (str): solver to use in MMPS iterations
        - max_iter (int): maximum number of iterations to do when solving the system in MMPS
        - tol (float): numerical tolerance used in the MMPS solver
    Returns
        - sample (xarray.Dataset): a sample drawn from p(x_{k} | x_{k-1}^{(i)}, y_{k})
    """
    # Instanciate a denoiser
    denoiser = ConditionalDenoiser(
        mask=mask,
        observed_variables=observed_variables,
        sigma_y=sigma_y,
        std_z=std_z,
        std_x=std_x,
        mean_x=mean_x,
        task_config=task_config,
        denoiser_architecture_config=denoiser_config,
        noise_encoder_config=noise_encoder_config,
        solver=solver,
        max_iter=max_iter,
        tol=tol,
    )

    # Instanciate a sampler
    if sampler == "dpm":
        _sampler = Samplers.DPM_Sampler(denoiser=denoiser, sampler_config=sampler_config)
    elif sampler == "ddim":
        _sampler = Samplers.DDIM_Sampler(denoiser=denoiser, **sampler_config)
    elif sampler == "abs":
        _sampler = Samplers.ABSampler(denoiser=denoiser, **sampler_config)
    else:
        raise ValueError(f"Unknown sampler «{sampler}». Choose between 'dpm', 'ddim' and 'abs'.")

    # Instanciate a predictor
    predictor = Predictor(
        std_z=std_z,
        min_x=min_x,
        std_x=std_x,
        mean_x=mean_x,
        sampler=_sampler,
    )

    # Extract observations from the reference
    observations = reference[observed_variables]
    observations = utils.convert_xarray_to_jax(observations, False)
    observations = jnp.array(observations)
    observations = observations[:, mask, :]
    observations = observations.reshape((
        observations.shape[0],
        -1,
    ))

    # Use the sampling function of the conditional sampler
    return predictor(
        inputs=inputs,
        target_template=target_template,
        forcings=forcings,
        observations=observations,
    )


# Jitted version of the function
conditional_sampling_jitted = jax.jit(
    lambda rng, i, t, f, r: conditional_sampling.apply(
        ckpt.params,
        {},
        rng,
        inputs=i,
        target_template=t,
        forcings=f,
        task_config=ckpt.task_config,
        denoiser_config=denoiser_architecture_config,
        noise_encoder_config=ckpt.noise_encoder_config,
        sampler="abs",
        sampler_config=sampler_config,
        min_x=min_x,
        std_x=std_x,
        std_z=std_z,
        mean_x=mean_x,
        reference=r,
        mask=mask,
        observed_variables=observed_variables,
        sigma_y=sigma_y,
        solver=solver,
        max_iter=max_iter,
        tol=tol,
    )[0]
)

# Loop on the number of sample to generate (while x8 GPU architectures are not available)
path = "./data/PPC/"
num_samples = 200
for i in tqdm.tqdm(range(1, num_samples + 1)):
    # Draw a conditional sample
    rng = jax.random.PRNGKey(i)
    sample = conditional_sampling_jitted(
        rng,
        eval_inputs,
        eval_targets,
        eval_forcings,
        reference,
    )

    # Save the sample
    file_name = path + "sample_" + str(i) + ".nc"
    sample.to_netcdf(file_name)

    # Free memory
    del sample
    gc.collect()
    jax.clear_caches()


"""
# Pmapped version for running in parallel
conditional_sampling_pmap = xarray_jax.pmap(conditional_sampling_jitted, dim="sample")

# Duplicate input data
inputs_batch = utils.duplicate_xarray(eval_inputs, new_dim="sample", n=1)
targets_batch = utils.duplicate_xarray(eval_targets, new_dim="sample", n=1)
forcings_batch = utils.duplicate_xarray(eval_forcings, new_dim="sample", n=1)
reference_batch = utils.duplicate_xarray(reference, new_dim="sample", n=1)

# Generate samples on multiple GPUs
samples = conditional_sampling_pmap(
    inputs_batch,
    targets_batch,
    forcings_batch,
    reference_batch,
)
"""

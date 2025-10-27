# Libraries
import dataclasses
import gc
import haiku as hk
import jax
import jax.numpy as jnp  # type: ignore
import numpy as np
import os
import xarray

from jax import Array  # type: ignore
from tqdm import tqdm
from typing import Dict, List, Union

from .wrapper import utils
from .wrapper.denoisers import ConditionalDenoiser, GenCastDenoiser
from .wrapper.graphcast import (
    checkpoint,
    data_utils,
    denoiser,
    gencast,
    graphcast,
    samplers_utils,
    xarray_jax,  # noqa: F401
)
from .wrapper.predictor import Predictor
from .wrapper.sampler import ABSampler, DDIM_Sampler, DPM_Sampler


@hk.transform_with_state
def unconditional_sampling(
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
) -> xarray.Dataset:
    """
    Draw a sample from p(x^{k+1} | x^{k}).
    Input(s)
        - inputs (xarray.Dataset): unnormalized previous states of the system with dimensions (batch=1, time=2, lat=181, lon=360, levels=13)
        - target_template (xarray.Dataset): template for the output with dimensions (batch=1, time=1, lat=181, lon=360, levels=13)
        - forcings (xarray.Dataset): unnormalized forcings terms used by the GenCast denoiser with dimensions (batch=1, time=1, lat=181, lon=360, levels=13)
        - task_config (graphcast.TaskConfig)
        - denoiser_config (denoiser.DenoiserArchitectureConfig)
        - noise_encoder_config (denoiser.NoiseEncoderConfig)
        - sampler (str): sampler to use
        - sampler_config (Union[Any, gencast.SamplerConfig])
        - min_x (xarray.Dataset): minimum values of system states for each variable
        - std_x (xarray.Dataset): standard deviation of system states for each variable
        - std_z (xarray.Dataset): standard deviation of residuals for each variable
        - mean_x (xarray.Dataset): mean of system states for each variable
    Returns
        - sample (xarray.Dataset): a sample drawn from p(x^{k+1} | x^{k})
    """
    # Instanciate a classical GenCast denoiser
    denoiser = GenCastDenoiser(
        task_config=task_config,
        denoiser_architecture_config=denoiser_config,
        noise_encoder_config=noise_encoder_config,
    )

    # Instanciate a sampler
    if sampler == "dpm":
        _sampler = DPM_Sampler(denoiser=denoiser, sampler_config=sampler_config)
    elif sampler == "ddim":
        _sampler = DDIM_Sampler(denoiser=denoiser, **sampler_config)
    elif sampler == "abs":
        _sampler = ABSampler(denoiser=denoiser, **sampler_config)
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

    # Use the predictor to generate a sample
    return predictor(
        inputs=inputs,
        target_template=target_template,
        forcings=forcings,
        observations=None,
    )


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
    reference: Array,
    sigma_y: Array,
    mask_satellite: Union[Array, None],
    mask_weather_stations: Union[Array, None],
    observed_variables_satellite: Union[List[str], None],
    observed_variables_weather_stations: Union[List[str], None],
    solver: str = None,
    max_iter: int = None,
    tol: float = None,
) -> xarray.Dataset:
    """
    Draw a sample from p(x^{k+1} | x^{k}, y^{k+1}).
    Input(s)
        - inputs (xarray.Dataset): unnormalized previous states of the system with dimensions (batch=1, time=2, lat=181, lon=360, levels=13)
        - target_template (xarray.Dataset): template for the output with dimensions (batch=1, time=1, lat=181, lon=360, levels=13)
        - forcings (xarray.Dataset): unnormalized forcings terms used by the GenCast denoiser with dimensions (batch=1, time=1, lat=181, lon=360, levels=13)
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
        - mask_satellite (Union[Array, None]): boolean Array of dimension (181, 360) corresponding to satellite observations
        - mask_weather_stations (Union[Array, None]): boolean Array of dimension (181, 360) corresponding to ground observations
        - observed_variables_satellite (Union[List[str], None]): ordered list of variables observed by satellite
        - observed_variables_weather_stations (Union[List[str], None]): ordered list of variables observed by ground weather stations
        - sigma_y (Array): covariance matrix of normalized observations with dimension (1, num_observed_variables)
        - solver (str): solver to use in MMPS iterations
        - max_iter (int): maximum number of iterations to do when solving the system in MMPS
        - tol (float): numerical tolerance used in the MMPS solver
    Returns
        - sample (xarray.Dataset): a sample drawn from p(x^{k+1} | x^{k}, y^{k+1})
    """
    # Instanciate an MMPS denoiser
    denoiser = ConditionalDenoiser(
        mask_satellite=mask_satellite,
        mask_weather_stations=mask_weather_stations,
        observed_variables_satellite=observed_variables_satellite,
        observed_variables_weather_stations=observed_variables_weather_stations,
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
        _sampler = DPM_Sampler(denoiser=denoiser, sampler_config=sampler_config)
    elif sampler == "ddim":
        _sampler = DDIM_Sampler(denoiser=denoiser, **sampler_config)
    elif sampler == "abs":
        _sampler = ABSampler(denoiser=denoiser, **sampler_config)
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

    # Clean and normalize observations
    variable_to_clean = "sea_surface_temperature"
    if variable_to_clean in inputs.keys():
        clean_reference = utils.clean_NaN(reference, variable_to_clean, min_x[variable_to_clean])
    else:
        clean_reference = reference
    normalized_reference = utils.normalize(clean_reference, std_x, mean_x)

    # Extract observation from weather stations
    if (observed_variables_weather_stations is not None) and (mask_weather_stations is not None):
        obs_weather_stations = normalized_reference[observed_variables_weather_stations]
        obs_weather_stations = utils.convert_xarray_to_jax(obs_weather_stations)
        obs_weather_stations = obs_weather_stations[:, mask_weather_stations, :]
        obs_weather_stations = obs_weather_stations.reshape((
            obs_weather_stations.shape[0],
            -1,
        ))
    else:
        obs_weather_stations = jnp.array([[]])

    # Extract observation from satellite
    if (observed_variables_satellite is not None) and (mask_satellite is not None):
        obs_satellite = normalized_reference[observed_variables_satellite]
        obs_satellite = utils.convert_xarray_to_jax(obs_satellite)
        obs_satellite = obs_satellite[:, mask_satellite, :]
        obs_satellite = obs_satellite.reshape((
            obs_satellite.shape[0],
            -1,
        ))
    else:
        obs_satellite = jnp.array([[]])

    # Concatenate observations from ground stations and satellite
    observations = jnp.concatenate([obs_weather_stations, obs_satellite], axis=1)

    # Use the sampling function of the conditional sampler
    return predictor(
        inputs=inputs,
        target_template=target_template,
        forcings=forcings,
        observations=observations,
    )


def ppc(
    num_samples: int,
    conditional_output_path: str,
    unconditional_output_path: str,
    data_path: str,
    checkpoint_path: str,
    min_x_path: str,
    std_x_path: str,
    std_z_path: str,
    mean_x_path: str,
    sampler: str,
    sampler_config: Union[Dict, gencast.SamplerConfig],
    mask_sat_path: str,
    mask_ws_path: str,
    observed_variables_sat: List[str],
    observed_variables_ws: List[str],
    sigma_y_sat_path: Array,
    sigma_y_ws_path: Array,
    solver: str,
    max_iter: int,
    tol: float,
):
    """
    Draw unconditional and conditional samples to latter generate observations with them.
    Input(s)
        - num_samples (int): number of samples to generate
        - conditional_output_path (str): path to the folder where conditional samples are stored
        - unconditional_output_path (str): path to the folder where unconditional samples are stored
        - data_path (str): path to the input data (x^{k} and y^{k+1})
        - min_x_path (str): path to min_x statistic
        - std_x_path (str): path to std_x statistic
        - std_z_path (str): path to std_z statistic
        - mean_x_path (str): path to mean_x statistic
        - sampler (str): sampler to use during the reverse diffusion process
        - sampler_config (Union[Dict, gencast.SamplerConfig]): configuration of the sampler
        - mask_sat_path (str): path to satellite mask
        - mask_ws_path (str): path to ground weather stations mask
        - observed_variables_sat (str): ordered list of variables observed by satellite
        - observed_variables_ws (str): ordered list of variables observed by ground weather stations
        - sigma_y_sat_path (str): path to the covariance matrix of unnormalized satellite observations (with dimension (len(observed_variables_sat), 13)))
        - sigma_y_ws_path (str): path to the covariance matrix of unnormalized ground observations (with dimension (len(observed_variables_ws),)
        - solver (str): solver to use in MMPS iterations
        - max_iter (int): maximum number of iterations to do when solving the system in MMPS
        - tol (float): numerical tolerance used in the MMPS solver
    """
    # Load the checkpoint
    with open(checkpoint_path, "rb") as file:
        ckpt = checkpoint.load(file, gencast.CheckPoint)

    # Load statistics
    with open(std_x_path, "rb") as file:
        std_x = xarray.load_dataset(file, decode_timedelta=True).compute()
    with open(std_z_path, "rb") as file:
        std_z = xarray.load_dataset(file, decode_timedelta=True).compute()
    with open(mean_x_path, "rb") as file:
        mean_x = xarray.load_dataset(file, decode_timedelta=True).compute()
    with open(min_x_path, "rb") as file:
        min_x = xarray.load_dataset(file, decode_timedelta=True).compute()

    # Inputs, targets and forcings
    with open(data_path, "rb") as file:
        data = xarray.load_dataset(file, decode_timedelta=True).compute()
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        data,
        target_lead_times=slice("12h", f"{(data.sizes['time'] - 2) * 12}h"),
        **dataclasses.asdict(ckpt.task_config),
    )
    eval_targets = eval_targets.isel(time=[0])
    eval_forcings = eval_forcings.isel(time=[0])
    del data
    gc.collect()

    # Prepare the sampler config
    if sampler == "dpm":
        sampler_config = ckpt.sampler_config
    else:
        noise_levels = samplers_utils.noise_schedule(
            max_noise_level=sampler_config["max_noise_level"],
            min_noise_level=sampler_config["min_noise_level"],
            num_noise_levels=sampler_config["num_noise_levels"],
            rho=sampler_config["rho"],
        )
        sampler_config["noise_levels"] = noise_levels
        _ = sampler_config.pop("max_noise_level")
        _ = sampler_config.pop("min_noise_level")
        _ = sampler_config.pop("num_noise_levels")
        _ = sampler_config.pop("rho")

    # Modify denoiser configuration for GPU
    denoiser_architecture_config = ckpt.denoiser_architecture_config
    denoiser_architecture_config.sparse_transformer_config.mask_type = "full"
    denoiser_architecture_config.sparse_transformer_config.attention_type = "triblockdiag_mha"

    # Load masks
    mask_sat = jnp.array(np.load(mask_sat_path).astype(bool))
    if len(mask_sat.shape) == 3:
        mask_sat = mask_sat[0, :]
    mask_ws = jnp.array(np.load(mask_ws_path).astype(bool))

    # Load unnormalized covariance matrix
    sigma_y_sat = jnp.array(np.load(sigma_y_sat_path).astype(jnp.float32))
    sigma_y_ws = jnp.array(np.load(sigma_y_ws_path).astype(jnp.float32))

    # Normalized observations covariance matrix
    sigma_hat_y = utils.normalized_observation_covariance(
        std_x=std_x,
        mask_satellite=mask_sat,
        mask_weather_stations=mask_ws,
        sigma_y_satellite=sigma_y_sat,
        sigma_y_weather_stations=sigma_y_ws,
        observed_variables_satellite=observed_variables_sat,
        observed_variables_weather_stations=observed_variables_ws,
    )

    # Jitted version of the functions
    unconditional_sampling_jitted = jax.jit(
        lambda rng, i: unconditional_sampling.apply(
            ckpt.params,
            {},
            rng,
            inputs=i,
            target_template=eval_targets,
            forcings=eval_forcings,
            task_config=ckpt.task_config,
            denoiser_config=denoiser_architecture_config,
            noise_encoder_config=ckpt.noise_encoder_config,
            sampler=sampler,
            sampler_config=sampler_config,
            min_x=min_x,
            std_x=std_x,
            std_z=std_z,
            mean_x=mean_x,
        )[0]
    )

    conditional_sampling_jitted = jax.jit(
        lambda rng, i: conditional_sampling.apply(
            ckpt.params,
            {},
            rng,
            inputs=i,
            target_template=eval_targets,
            forcings=eval_forcings,
            task_config=ckpt.task_config,
            denoiser_config=denoiser_architecture_config,
            noise_encoder_config=ckpt.noise_encoder_config,
            sampler=sampler,
            sampler_config=sampler_config,
            min_x=min_x,
            std_x=std_x,
            std_z=std_z,
            mean_x=mean_x,
            reference=eval_targets,
            mask_satellite=mask_sat,
            mask_weather_stations=mask_ws,
            observed_variables_satellite=observed_variables_sat,
            observed_variables_weather_stations=observed_variables_ws,
            sigma_y=sigma_hat_y,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
        )[0]
    )

    # Pmapped version for running in parallel
    unconditional_sampling_pmap = xarray_jax.pmap(unconditional_sampling_jitted, dim="sample")
    conditional_sampling_pmap = xarray_jax.pmap(conditional_sampling_jitted, dim="sample")

    # Parallel sampling parameters
    num_gpus = len([device for device in jax.devices() if device.platform == "gpu"])
    assert int(num_samples % num_gpus) == 0
    num_steps = int(num_samples // num_gpus)

    # Duplicate the inputs to get a batch for running in parallel
    input_batch = utils.duplicate_xarray(
        array=eval_inputs,
        new_dim="sample",
        n=num_gpus,
    )

    # Draw unconditional samples in parallel
    current_num_samples = int(
        sum(1 for f in os.listdir(unconditional_output_path) if f.endswith(".nc"))
    )
    if current_num_samples < num_samples:
        print("Draw unconditional samples...")
        count = 1
        for _ in tqdm(range(1, num_steps + 1)):
            # Sampling
            key = jax.random.PRNGKey(np.random.randint(100_000))
            keys = jax.random.split(key, num_gpus)
            samples = unconditional_sampling_pmap(keys, input_batch)

            # Save the samples
            for j in range(samples.sizes["sample"]):
                sample = samples.isel(sample=j)
                if unconditional_output_path[-1] == "/":
                    file_name = unconditional_output_path + str(count) + str(".nc")
                else:
                    file_name = unconditional_output_path + str("/") + str(count) + str(".nc")
                sample.to_netcdf(file_name)
                count += 1

            # Free memory
            del samples
            del sample
            del key
            del keys
            gc.collect()
            jax.clear_caches()

    # Draw conditional samples in parallel
    current_num_samples = int(
        sum(1 for f in os.listdir(conditional_output_path) if f.endswith(".nc"))
    )
    if current_num_samples < num_samples:
        print("Draw conditional samples...")
        count = 1
        for _ in tqdm(range(1, num_steps + 1)):
            # Sampling
            key = jax.random.PRNGKey(np.random.randint(100_000))
            keys = jax.random.split(key, num_gpus)
            samples = conditional_sampling_pmap(keys, input_batch)

            # Save the samples
            for j in range(samples.sizes["sample"]):
                sample = samples.isel(sample=j)
                if conditional_output_path[-1] == "/":
                    file_name = conditional_output_path + str(count) + str(".nc")
                else:
                    file_name = conditional_output_path + str("/") + str(count) + str(".nc")
                sample.to_netcdf(file_name)
                count += 1

            # Free memory
            del samples
            del sample
            del key
            del keys
            gc.collect()
            jax.clear_caches()

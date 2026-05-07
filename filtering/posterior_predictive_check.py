# Libraries
import dataclasses
import gc
import haiku as hk
import jax
import jax.numpy as jnp  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import xarray

from jax import Array  # type: ignore
from tqdm import tqdm
from typing import Dict, List, Tuple, Union

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
    coarsening_factor: Union[int, None],
    observed_variables_satellite: Union[List[str], None],
    observed_variables_weather_stations: Union[List[str], None],
    observed_variables_coarsening: Union[List[str], None],
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
        - observed_variables_coarsening (Union[List[str], None]): ordered list of observed variables for caorsening
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
        coarsening_factor=coarsening_factor,
        observed_variables_satellite=observed_variables_satellite,
        observed_variables_weather_stations=observed_variables_weather_stations,
        observed_variables_coarsening=observed_variables_coarsening,
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

    # Extract coarsened observations
    if (coarsening_factor is not None) and (observed_variables_coarsening is not None):
        obs_coarsening = normalized_reference[observed_variables_coarsening]
        obs_coarsening = utils.coarsening(x=obs_coarsening, factor=coarsening_factor)
        obs_coarsening = obs_coarsening.reshape(obs_coarsening.shape[0], -1)
        observations = obs_coarsening

    else:
        # Extract observation from weather stations
        if (observed_variables_weather_stations is not None) and (
            mask_weather_stations is not None
        ):
            obs_weather_stations = normalized_reference[observed_variables_weather_stations]
            obs_weather_stations = utils.convert_xarray_to_jax(
                obs_weather_stations, jax_array=False
            )
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
            obs_satellite = utils.convert_xarray_to_jax(obs_satellite, jax_array=False)
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
    mask_sat_path: Union[str, None],
    mask_ws_path: Union[str, None],
    coarsening_factor: Union[int, None],
    observed_variables_sat: Union[List[str], None],
    observed_variables_ws: Union[List[str], None],
    observed_variables_coarsening: Union[List[str], None],
    sigma_y_sat_path: Union[str, None],
    sigma_y_ws_path: Union[str, None],
    sigma_y_coarsening_path: Union[str, None],
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
        - mask_sat_path (Union[str, None]): path to satellite mask
        - mask_ws_path (Union[str, None]): path to ground weather stations mask
        - coarsening_factor (Union[int, None]): coarsening factor
        - observed_variables_sat (str): ordered list of variables observed by satellite
        - observed_variables_ws (str): ordered list of variables observed by ground weather stations
        - observed_variables_coarsening (Union[List[str], None]): ordered list of observed variables for coarsening
        - sigma_y_sat_path (str): path to the covariance matrix of unnormalized satellite observations (with dimension (len(observed_variables_sat), 13)))
        - sigma_y_ws_path (str): path to the covariance matrix of unnormalized ground observations (with dimension (len(observed_variables_ws),)
        - sigma_y_coarsening_path (str): path to the covariance matrix of unnormalized coarsened observations (with dimension (num_variables,)
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
            max_noise_level=float(sampler_config["max_noise_level"]),
            min_noise_level=float(sampler_config["min_noise_level"]),
            num_noise_levels=int(sampler_config["num_noise_levels"]),
            rho=float(sampler_config["rho"]),
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
    if mask_sat_path is not None:
        mask_sat = jnp.array(np.load(mask_sat_path).astype(bool))
        if len(mask_sat.shape) == 3:
            mask_sat = mask_sat[0, :]
    else:
        mask_sat = None
    if mask_ws_path is not None:
        mask_ws = jnp.array(np.load(mask_ws_path).astype(bool))
    else:
        mask_ws = None

    # Load unnormalized covariance matrix
    if sigma_y_sat_path is not None:
        sigma_y_sat = jnp.array(np.load(sigma_y_sat_path).astype(jnp.float32))
    else:
        sigma_y_sat = None
    if sigma_y_ws_path is not None:
        sigma_y_ws = jnp.array(np.load(sigma_y_ws_path).astype(jnp.float32))
    else:
        sigma_y_ws = None
    if sigma_y_coarsening_path is not None:
        sigma_y_coarsening = jnp.array(np.load(sigma_y_coarsening_path).astype(jnp.float32))
    else:
        sigma_y_coarsening = None

    # Normalized observations covariance matrix
    sigma_hat_y = utils.normalized_observation_covariance(
        std_x=std_x,
        mask_satellite=mask_sat,
        mask_weather_stations=mask_ws,
        coarsening_factor=coarsening_factor,
        sigma_y_satellite=sigma_y_sat,
        sigma_y_weather_stations=sigma_y_ws,
        sigma_y_coarsening=sigma_y_coarsening,
        observed_variables_satellite=observed_variables_sat,
        observed_variables_weather_stations=observed_variables_ws,
        observed_variables_coarsening=observed_variables_coarsening,
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
            coarsening_factor=coarsening_factor,
            observed_variables_satellite=observed_variables_sat,
            observed_variables_weather_stations=observed_variables_ws,
            observed_variables_coarsening=observed_variables_coarsening,
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
                sample.to_netcdf(file_name, engine="netcdf4", format="NETCDF4")
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
                sample.to_netcdf(file_name, engine="netcdf4", format="NETCDF4")
                count += 1

            # Free memory
            del samples
            del sample
            del key
            del keys
            gc.collect()
            jax.clear_caches()


def generate_distribution(
    eval_target: xarray.Dataset,
    conditional_path: str,
    unconditional_path: str,
    num_samples: int,
    num_draws: int,
    variables: List[Union[str, List]],
    lat: int,
    lon: int,
    mask_satellite: Array,
    mask_weather_stations: Array,
    stds: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an approximation of the conditional posterior predictive distribution for a specific variable at a specific location.
    Input(s)
        - eval_target (xarray.Dataset): xarray from which observations has been taken during posterior sampling
        - conditional_path (str): path to samples generated using posterior sampling
        - unconditional_path (str): path to samples genereted using the classical GenCast denoiser
        - num_samples (int): number of samples to use to approximate the conditional PPC disitribution
        - num_draws (int): number of samples to draw in order to approximate p(tilde{y}^{k+1} | x^{k+1}_{(i)})
        - variables (List[Union[str, List]]): variables of interest (and corresponding level of interest if it is atmospheric variable)
        - lat (int): latitude of interest, should be between -90 and 90
        - lon (int): longitude of interest, should be between 0 and 359
        - mask_satellite (Array): boolean Array of dimension (181, 360) corresponding to satellite observations
        - mask_weather_stations (Array): boolean Array of dimension (181, 360) corresponding to ground stations observations
        - stds (Dict): standard deviation of unnormalized observations for the variable of interest
    Returns
        - y (np.ndarray): y^{k+1} used by the conditional denoiser during posterior sampling with dimension (len(variables),)
        - distributions (np.ndarray): approximation of conditional and unconditional PPC distributions with dimension (2, len(variables), num_samples * num_draws)
    """
    # Check if there are enough samples
    num_conditional_samples = int(
        sum(1 for f in os.listdir(conditional_path) if f.endswith(".nc"))
    )
    num_unconditional_samples = int(
        sum(1 for f in os.listdir(unconditional_path) if f.endswith(".nc"))
    )
    assert num_conditional_samples >= num_samples
    assert num_unconditional_samples >= num_samples

    # Check latitude and longitude, and define the indice
    assert (-90 <= lat) and (lat <= 90)
    assert (0 <= lon) and (lon <= 359)
    lat, lon = int(lat + 90), lon

    # Check that the location has been observed during sampling
    for variable in variables:
        if isinstance(variable, List):
            assert bool(mask_satellite[lat, lon])
        else:
            assert bool(mask_weather_stations[lat, lon])

    # Approximate distributions
    distributions = []
    for i in tqdm(range(1, num_samples + 1)):
        # Load the conditional sample
        if conditional_path[-1] == "/":
            conditional_sample_path = conditional_path + str(i) + str(".nc")
        else:
            conditional_sample_path = conditional_path + str("/") + str(i) + str(".nc")
        with open(conditional_sample_path, "rb") as file:
            conditional_sample = xarray.load_dataset(file, decode_timedelta=True).compute()
        conditional_sample = conditional_sample.isel(lat=[lat], lon=[lon])

        # Load the unconditional sample
        if unconditional_path[-1] == "/":
            unconditional_sample_path = unconditional_path + str(i) + str(".nc")
        else:
            unconditional_sample_path = unconditional_path + str("/") + str(i) + str(".nc")
        with open(unconditional_sample_path, "rb") as file:
            unconditional_sample = xarray.load_dataset(file, decode_timedelta=True).compute()
        unconditional_sample = unconditional_sample.isel(lat=[lat], lon=[lon])

        # Extract data for each variable, add noise and update the lists
        conditional_distribution = []
        unconditional_distribution = []
        for variable in variables:
            if isinstance(variable, List):
                conditional_data = conditional_sample.sel(level=[int(variable[-1])])
                conditional_data = conditional_data[str(variable[0])].values.item()
                unconditional_data = unconditional_sample.sel(level=[int(variable[-1])])
                unconditional_data = unconditional_data[str(variable[0])].values.item()
            else:
                conditional_data = conditional_sample[str(variable)].values.item()
                unconditional_data = unconditional_sample[str(variable)].values.item()
            if isinstance(variable, List):
                std = float(stds[str(variable[0])][int(variable[-1])])
            else:
                std = float(stds[str(variable)])
            noise = std * np.random.randn(num_draws)
            conditional_data = conditional_data + noise
            unconditional_data = unconditional_data + noise

            # Update lists
            conditional_distribution.append(conditional_data)
            unconditional_distribution.append(unconditional_data)

        # Transform the lists to numpy arrays
        conditional_distribution = np.vstack(conditional_distribution)
        unconditional_distribution = np.vstack(unconditional_distribution)
        distribution = np.stack([conditional_distribution, unconditional_distribution], axis=0)
        distributions.append(distribution)

    # Convert the result to a numpy array
    distributions = np.concatenate(distributions, axis=-1)

    # Get the observation used during posterior sampling
    y = []
    eval_target = eval_target.isel(lat=[lat], lon=[lon])
    for variable in variables:
        if isinstance(variable, List):
            obs = eval_target.sel(level=[int(variable[-1])])
            obs = obs[str(variable[0])].values.item()
        else:
            obs = eval_target[str(variable)].values.item()
        y.append(obs)
    y = np.asarray(y)

    return y, distributions


def plot_PPC(
    reference_path: str,
    checkpoint_path: str,
    conditional_path: str,
    unconditional_path: str,
    output_path: str,
    mask_sat_path: str,
    mask_ws_path: str,
    variables: List,
    stds: Dict,
    lat: int,
    lon: int,
    num_samples: int,
    num_draws: int,
    num_row: int,
    num_col: int,
    title: str,
    figsize: Tuple[int],
    colors: List[str],
    xlabels: List[str],
):
    """
    Plot the result of the PPC
    Input(s)
        - reference_path (str): path to the reference data from which x^{k} has been taken
        - checkpoint_path (str): path to the checkpoint needed to load the reference data
        - conditional_path (str): path to conditional samples
        - unconditional_path (str): path to unconditional samples
        - output_path (str): output path to save the image
        - mask_sat_path (str): path to the satellite mask used during posterior sampling
        - mask_ws_path (str): path to ground weather stations mask used during posterior sampling
        - variables (List): list of variables to plot
        - stds (Dict): dictionary containing standard deviation of variables to plot
        - lat (int): latitude of interest, should be between -90 and 90
        - lon (int): latitude of interest, should be between 0 and 359
        - num_samples (int): number of samples to use to approximate the conditional PPC disitribution
        - num_draws (int): number of samples to draw in order to approximate p(tilde{y}^{k+1} | x^{k+1}_{(i)})
        - num_row (int): number of rows in the figure
        - num_col (int): number of columns in the figure
        - title (str): global title of the figure
        - figsize (Tuple[int]): size of the figure
        - colors (List[str]): colors to use for the figure
        - xlabels (List[str]): labels to use for the figure
    """
    # Checks
    assert len(variables) == (num_row * num_col)

    # Load the eval_target
    with open(checkpoint_path, "rb") as file:
        ckpt = checkpoint.load(file, gencast.CheckPoint)
    with open(reference_path, "rb") as file:
        data = xarray.load_dataset(file, decode_timedelta=True).compute()
    _, eval_targets, _ = data_utils.extract_inputs_targets_forcings(
        data,
        target_lead_times=slice("12h", f"{(data.sizes['time'] - 2) * 12}h"),
        **dataclasses.asdict(ckpt.task_config),
    )
    eval_targets = eval_targets.isel(time=[0])
    del ckpt, data
    gc.collect()

    # Load the mask
    mask_sat = jnp.array(np.load(mask_sat_path).astype(bool))
    if len(mask_sat.shape) == 3:
        mask_sat = mask_sat[0, :]
    mask_ws = jnp.array(np.load(mask_ws_path).astype(bool))

    # Create the figure
    fig, axes = plt.subplots(num_row, num_col, figsize=figsize)
    axes = axes.flatten()

    # Get the data to be plotted
    y, distributions = generate_distribution(
        eval_target=eval_targets,
        conditional_path=conditional_path,
        unconditional_path=unconditional_path,
        num_samples=num_samples,
        num_draws=num_draws,
        variables=variables,
        lat=lat,
        lon=lon,
        mask_satellite=mask_sat,
        mask_weather_stations=mask_ws,
        stds=stds,
    )

    # Create subplots
    for j, ax in enumerate(axes):
        # Plot the conditional data
        sns.kdeplot(
            distributions[0, j, :],
            ax=ax,
            color=colors[0],
            label=r"$q(\tilde{y}^{k+1} \mid x^{k+1}, y^{k+1})$",
            fill=True,
            alpha=0.5,
        )

        # Plot the unconditional data
        sns.kdeplot(
            distributions[1, j, :],
            ax=ax,
            color=colors[1],
            label=r"$q(\tilde{y}^{k+1} \mid x^{k+1})$",
            fill=True,
            alpha=0.5,
        )

        # Plot the true observation
        ax.axvline(y[j], color=colors[-1], linestyle="--", label=r"$y^{k+1}$")

        # Set the label
        ax.set_xlabel(xlabels[j], fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=10)

    # Global title
    plt.suptitle(title, fontsize=14, y=0.95)

    # Adjustements
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    sns.set_style("whitegrid")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

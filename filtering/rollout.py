# Libraries
import dataclasses
import gc
import haiku as hk
import jax  # type: ignore
import numpy as np
import xarray

from pathlib import Path
from tqdm import tqdm
from typing import Dict, Union

from .wrapper.denoisers import GenCastDenoiser
from .wrapper.graphcast import (
    checkpoint,
    data_utils,
    denoiser,
    gencast,
    graphcast,
    samplers_utils,
    xarray_jax,
)
from .wrapper.predictor import Predictor
from .wrapper.sampler import (
    ABSampler,
    DDIM_Sampler,
    DPM_Sampler,
)


@hk.transform_with_state
def step(
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
    Draw next state from p(x^{k+1} | x^{k})
    Input(s)
        - inputs (xarray.Dataset): unnormalized previous state x^{k} with dimension (batch=1, time=2, lat=181, lon=360, levels=13)
        - target_template (xarray.Dataset): target template with dimension (batch=1, time=1, lat=181, lon=360, levels=13)
        - forcings (xarray.Dataset): unnormalized forcing terms used the GenCast denoiser with dimension (batch=1, time=1, lat=181, lon=360, levels=13)
        - task_config (graphcast.TaskConfig)
        - denoiser_config (denoiser.DenoiserArchitectureConfig)
        - noise_encoder_config (denoiser.NoiseEncoderConfig)
        - sampler (str): sampler to use
        - sampler_config (Union[Dict, gencast.SamplerConfig])
        - min_x (xarray.Dataset): minimum values of system states for each variable
        - std_x (xarray.Dataset): standard deviation of system states for each variable
        - std_z (xarray.Dataset): standard deviation of residuals for each variable
        - mean_x (xarray.Dataset): mean of system states for each variable
    Returns
        - next_state (xarray.Dataset): next state x^{k+1} with dimension (batch=1, time=2, lat=181, lon=360, levels=13)
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

    # Use the sampling function of the conditional sampler
    next_state = predictor(
        inputs=inputs,
        target_template=target_template,
        forcings=forcings,
    )

    # Update the state for the next step
    next_state = xarray.merge([next_state, forcings])
    next_state = next_state.drop_vars("total_precipitation_12hr")
    next_state = xarray.concat([inputs, next_state], dim="time", data_vars="minimal")
    next_state = next_state.isel(time=slice(-2, None))

    return next_state


def generate_trajectories(
    num_samples: int,
    output_path: str,
    data_path: str,
    checkpoint_path: str,
    sampler: str,
    sampler_config: Union[Dict, gencast.SamplerConfig],
    min_x_path: xarray.Dataset,
    std_x_path: xarray.Dataset,
    std_z_path: xarray.Dataset,
    mean_x_path: xarray.Dataset,
):
    """
    Generate one complete trajectory composed of n times steps
    Input(s)
        - num_samples (int): number of unconditional samples/trajectories to generate
        - output_path (str): path where samples/trajectories are saved
        - data_path (str): path to the data to get initial condition, target template and forcings
        - checkpoint_path (str): path to the checkpoint
        - sampler (str): sampler to use during the reverse diffusion process
        - sampler_config (Union[Dict, gencast.SamplerConfig]): configuration of the sampler
        - min_x_path (str): path to min_x statistic
        - std_x_path (str): path to std_x statistic
        - std_z_path (str): path to std_z statistic
        - mean_x_path (str): path to mean_x statistic
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
    x0, targets, forcings = data_utils.extract_inputs_targets_forcings(
        data,
        target_lead_times=slice("12h", f"{(data.sizes['time'] - 2) * 12}h"),
        **dataclasses.asdict(ckpt.task_config),
    )
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

    # Get the number of steps to do
    num_steps = targets.sizes["time"]
    assert num_steps == forcings.sizes["time"]

    # Parallel sampling parameters
    num_gpus = len([device for device in jax.devices() if device.platform == "gpu"])
    assert int(num_samples % num_gpus) == 0
    num_batch = int(num_samples // num_gpus)

    # Duplicate initial conditions
    if output_path[-1] == "/":
        ic_folder = Path(output_path + "0/")
    else:
        ic_folder = Path(output_path + "/0/")
    ic_folder.mkdir(parents=True, exist_ok=True)
    for i in range(1, num_samples + 1):
        if output_path[-1] == "/":
            file_name = output_path + str("0/") + str(i) + str(".nc")
        else:
            file_name = output_path + str("/0/") + str(i) + str(".nc")
        x0.to_netcdf(file_name, format="NETCDF4", engine="netcdf4")

    # Loop on the number of simulation steps to do
    for i in range(1, num_steps + 1):
        # Define previous and new particles path
        print("Step n°{}".format(i))
        if output_path[-1] == "/":
            previous_particles_path = output_path + str(i - 1) + str("/")
            new_particles_path = output_path + str(i) + str("/")
        else:
            previous_particles_path = output_path + str("/") + str(i - 1) + str("/")
            new_particles_path = output_path + str("/") + str(i) + str("/")

        # Create a folder for the new particles (if it does not exist)
        new_folder = Path(new_particles_path)
        new_folder.mkdir(parents=True, exist_ok=True)

        # Count the number of files
        num_existing_samples = list(new_folder.glob("*.nc"))
        if len(num_existing_samples) >= num_samples:
            print(
                f"→ {num_samples} particles are already present for step {i}, moving directly to step {i + 1}!"
            )
        else:
            # Get current forcings and template
            current_forcings = forcings.isel(time=[int(i - 1)])
            current_template = targets.isel(time=[int(i - 1)])

            # Define a jitted step function
            step_jitted = jax.jit(
                lambda rng, i: step.apply(
                    ckpt.params,
                    {},
                    rng,
                    inputs=i,
                    target_template=current_template,  # noqa: B023
                    forcings=current_forcings,  # noqa: B023
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

            # Define a pmap version to run in parallel
            step_pmap = xarray_jax.pmap(step_jitted, dim="sample")

            # Loop on the number steps
            print("     Draw samples...")
            count = 1
            for j in tqdm(range(1, num_batch + 1)):
                # Get a batch of particles to do the job in parallel
                samples, start_index = [], (j - 1) * num_gpus + 1
                for index in range(start_index, min(start_index + num_gpus, num_samples + 1)):
                    particle_path = previous_particles_path + str(index) + ".nc"
                    with open(particle_path, "rb") as file:
                        particle = xarray.load_dataset(file, decode_timedelta=True).compute()
                    samples.append(particle)

                # Do computations in parallel
                key = jax.random.PRNGKey(np.random.randint(100_000))
                keys = jax.random.split(key, num_gpus)
                samples = xarray.concat(
                    samples, dim=xarray.DataArray([k for k in range(len(samples))], dims="sample")
                )
                next_samples = step_pmap(keys, samples)

                # Convert to a list
                next_samples = [
                    next_samples.isel(sample=k) for k in range(next_samples.sizes["sample"])
                ]

                # Update the inputs for next step and save it
                for _, next_sample in enumerate(next_samples):
                    file_name = new_particles_path + str(count) + ".nc"
                    next_sample.to_netcdf(file_name, format="NETCDF4", engine="netcdf4")
                    count += 1

                # Free memory
                del samples
                del next_samples
                gc.collect()
                jax.clear_caches()

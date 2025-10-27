# Libraries
import gc
import haiku as hk
import jax  # type: ignore
import warnings
import xarray

from pathlib import Path
from tqdm import tqdm
from typing import Dict, Union

from .wrapper.denoisers import GenCastDenoiser
from .wrapper.graphcast import (
    denoiser,
    gencast,
    graphcast,
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
    # Instanciate a denoiser
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
    N: int,
    traj_path: str,
    x0: xarray.Dataset,
    target_template: xarray.Dataset,
    forcings: xarray.Dataset,
    ckpt: gencast.CheckPoint,
    task_config: graphcast.TaskConfig,
    denoiser_config: denoiser.DenoiserArchitectureConfig,
    noise_encoder_config: denoiser.NoiseEncoderConfig,
    sampler: str,
    sampler_config: Union[Dict, gencast.SamplerConfig],
    min_x: xarray.Dataset,
    std_x: xarray.Dataset,
    std_z: xarray.Dataset,
    mean_x: xarray.Dataset,
):
    """
    Generate one complete trajectory composed of n times steps
    Input(s)
        - N (int): number of unconditional trajectories to generate
        - traj_path (str): path to the trajectories
        - x0 (xarray.Dataset): initial condition for the trajectories with dimension (batch=1, time=2, lat=181, lon=360, levels=13)
        - target_template (xarray.Dataset): templates with dimensions (batch=1, time=n, lat=181, lon=360, levels=13)
        - forcings (xarray.Dataset): forcings terms used by the GenCast denoiser with dimensions (batch=1, time=n, lat=181, lon=360, levels=13)
        - ckpt (gencast.CheckPoint): checkpoint to use (gencast_1deg.npz for example)
        - task_config (graphcast.TaskConfig)
        - denoiser_config (denoiser.DenoiserArchitectureConfig)
        - noise_encoder_config (denoiser.NoiseEncoderConfig)
        - sampler (str): sampler to use
        - sampler_config (Union[Any, gencast.SamplerConfig])
        - min_x (xarray.Dataset): minimum values of system states for each variable
        - std_x (xarray.Dataset): standard deviation of system states for each variable
        - std_z (xarray.Dataset): standard deviation of residuals for each variable
        - mean_x (xarray.Dataset): mean of system states for each variable
        - verbose (bool): if True, the generation process is monitored by tqdm
    Returns
        - chunks (xarray.Dataset): a trajectory with dimension (batch=1, time=n, lat=181, lon=360, levels=13)
    """
    # Get the number of steps to do
    num_steps = target_template.sizes["time"]
    assert num_steps == forcings.sizes["time"]

    # Duplicate initial conditions
    ic_folder = Path(traj_path + "/0/")
    ic_folder.mkdir(parents=True, exist_ok=True)
    for i in range(1, N + 1):
        file_name = traj_path + "/0/" + str(i) + ".nc"
        x0.to_netcdf(file_name)

    # Ignore warnings of GenCast (about sparsity)
    warnings.filterwarnings("ignore")

    # Loop on the number of steps
    for i in range(1, num_steps + 1):
        # Define previous and new particles path
        print("Step n°{}".format(i))
        previous_particles_path = traj_path + "/" + str(i - 1) + "/"
        new_particles_path = traj_path + "/" + str(i) + "/"

        # Create the new particles folder
        new_folder = Path(new_particles_path)
        new_folder.mkdir(parents=True, exist_ok=True)

        # Count the number of files
        existing_files = list(new_folder.glob("*.nc"))
        if len(existing_files) == N:
            print(
                f"→ {N} particles are already present for step {i}, moving directly to step {i + 1}!"
            )
        else:
            # Get current forcings and template
            current_forcings = forcings.isel(time=[i - 1])
            current_template = target_template.isel(time=[i - 1])

            # Define a jitted step function
            step_jitted = jax.jit(
                lambda rng, i: step.apply(
                    ckpt.params,
                    {},
                    rng,
                    inputs=i,
                    target_template=current_template,  # noqa: B023
                    forcings=current_forcings,  # noqa: B023
                    task_config=task_config,
                    denoiser_config=denoiser_config,
                    noise_encoder_config=noise_encoder_config,
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

            # Compute the number of steps to do
            num_devices = len(jax.devices())
            if N % num_devices == 0:
                num_steps = N // num_devices
            else:
                num_steps = N // num_devices + 1

            # Loop on the number steps
            print("     Draw samples...")
            count = 1
            for j in tqdm(range(1, num_steps + 1)):
                samples = []
                start_index = (j - 1) * num_devices + 1

                # Get a batch of particles to do the job in parallel
                for index in range(start_index, min(start_index + num_devices, N + 1)):
                    particle_path = previous_particles_path + str(index) + ".nc"
                    with open(particle_path, "rb") as file:
                        particle = xarray.load_dataset(file, decode_timedelta=True).compute()
                    samples.append(particle)

                # Do computations in parallel
                key = jax.random.PRNGKey(10_000 + 2 * j)
                keys = jax.random.split(key, num_devices)
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
                    next_sample.to_netcdf(file_name)
                    count += 1

                # Free memory
                del samples
                del next_samples
                gc.collect()
                jax.clear_caches()

# Libraries
import haiku as hk
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import warnings
import xarray

from jax import Array  # type: ignore
from tqdm import tqdm
from typing import Dict, List, Optional, Union

import sampler as Samplers
import utils

from denoisers import ConditionalDenoiser, GenCastDenoiser
from graphcast import denoiser, gencast, graphcast
from predictor import Predictor


def generate_trajectory(
    inputs: xarray.Dataset,
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
    seed: int,
    verbose: bool = True,
    reference: Optional[xarray.Dataset] = None,
    mask: Optional[Array] = None,
    observed_variables: Optional[List[str]] = None,
    sigma_y: Optional[Array] = None,
    solver: Optional[str] = None,
    max_iter: Optional[int] = None,
    tol: Optional[float] = None,
) -> xarray.Dataset:
    """
    Generate one complete trajectory composed of n times steps
    Input(s)
        - inputs (xarray.Dataset): previous states hat{x}_{t-2, t-1}^{(i)} of the system with dimensions (batch=1, time=2, lat=181, lon=360, levels=13)
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
        - seed (int): seed to use as input of jax.random.PRNGKey
        - verbose (bool): if True, the generation process is monitored by tqdm
        - reference (Optional[xarray.Dataset]): reference from which observations are extracted with dimensions (batch=1, time=n, lat=181, lon=360, levels=13)
        - mask (Optional[Array]): mask used to do subsampling with dimension (181, 360)
        - observed_variables (Optional[List[str]]): ordered list of observed variables
        - sigma_y (Optional[Array]): covariance matrix of observations Sigma_{y} with dimension (len(observed_variables),)
        - solver (Optional[str]): solver to use in MMPS iterations
        - max_iter (Optional[int]): maximum number of iterations to do when solving the system in MMPS
        - tol (Optional[float]): numerical tolerance used in the MMPS solver
    Returns
        - chunks (xarray.Dataset): a trajectory with dimension (batch=1, time=n, lat=181, lon=360, levels=13)
    """

    # Function to draw a plausible futur state of the system from p(x_{t} | x_{t-1}, x_{t-2}) or p(x_{t} | x_{t-1}, x_{t-2}, y_{t})
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
        observations: Optional[Array] = None,
        mask: Optional[Array] = None,
        observed_variables: Optional[List[str]] = None,
        sigma_y: Optional[Array] = None,
        solver: Optional[str] = None,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
    ) -> xarray.Dataset:
        # Instanciate a denoiser
        if observations is not None:
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
        else:
            denoiser = GenCastDenoiser(
                task_config=task_config,
                denoiser_architecture_config=denoiser_config,
                noise_encoder_config=noise_encoder_config,
            )

        # Instanciate a sampler
        if sampler == "dpm":
            _sampler = Samplers.DPM_Sampler(denoiser=denoiser, sampler_config=sampler_config)
        elif sampler == "ddim":
            _sampler = Samplers.DDIM_Sampler(denoiser=denoiser, **sampler_config)
        elif sampler == "abs":
            _sampler = Samplers.ABSampler(denoiser=denoiser, **sampler_config)
        else:
            raise ValueError(
                f"Unknown sampler «{sampler}». Choose between 'dpm', 'ddim' and 'abs'."
            )

        # Instanciate a predictor
        predictor = Predictor(
            std_z=std_z,
            min_x=min_x,
            std_x=std_x,
            mean_x=mean_x,
            sampler=_sampler,
        )

        # Use the sampling function of the conditional sampler
        return predictor(
            inputs=inputs,
            target_template=target_template,
            forcings=forcings,
            observations=observations,
        )

    # Jitted version of the function
    step_jitted = jax.jit(
        lambda rng, i, t, f, o: step.apply(
            ckpt.params,
            {},
            rng,
            inputs=i,
            target_template=t,
            forcings=f,
            task_config=task_config,
            denoiser_config=denoiser_config,
            noise_encoder_config=noise_encoder_config,
            sampler=sampler,
            sampler_config=sampler_config,
            min_x=min_x,
            std_x=std_x,
            std_z=std_z,
            mean_x=mean_x,
            observations=o,
            mask=mask,
            observed_variables=observed_variables,
            sigma_y=sigma_y,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
        )[0]
    )

    # Get the number of steps to do
    num_steps = target_template.sizes["time"]
    assert num_steps == forcings.sizes["time"]

    # Define a random key
    key = jax.random.PRNGKey(seed)

    # Ignore warnings of GenCast (about sparsity)
    warnings.filterwarnings("ignore")

    # Loop on the number of steps
    chunks = []
    iterator = tqdm(range(num_steps)) if verbose else range(num_steps)
    for k in iterator:
        # Get current forcings and template
        current_forcings = forcings.isel(time=[k])
        current_template = target_template.isel(time=[k])

        # Get current observation (if observations are given)
        if reference is not None:
            current_reference = reference.isel(time=[k])
            current_observations = current_reference[observed_variables]
            current_observations = utils.convert_xarray_to_jax(current_observations, False)
            current_observations = jnp.array(current_observations)
            current_observations = current_observations[:, mask, :]
            current_observations = current_observations.reshape((
                current_observations.shape[0],
                -1,
            ))
        else:
            current_observations = None

        # Use them to draw a plausible next state
        chunk = step_jitted(
            key,
            inputs,
            current_template,
            current_forcings,
            current_observations,
        )

        # Update the list of chunks
        chunks.append(chunk.copy(deep=True))

        # Update the inputs for next step
        chunk = xarray.merge([chunk, current_forcings])
        chunk = chunk.drop_vars("total_precipitation_12hr")
        inputs = xarray.concat([inputs, chunk], dim="time", data_vars="minimal")
        inputs = inputs.isel(time=slice(-2, None))

    # Convert chunks to an xarray
    chunks = xarray.concat(chunks, dim="time", data_vars="minimal")
    chunks.sortby("time")

    return chunks

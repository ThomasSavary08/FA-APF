# Libraries
import dataclasses
import gc
import haiku as hk
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import numpy as np
import xarray

from jax import Array  # type: ignore
from pathlib import Path
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
    xarray_jax,
)
from .wrapper.predictor import Predictor
from .wrapper.sampler import ABSampler, DDIM_Sampler, DPM_Sampler


def weighting(
    N: int,
    N_thr_min: int,
    N_thr_max: int,
    alpha_init: float,
    previous_particle_path: str,
    observations: Array,
    mask_sat: Union[Array, None],
    mask_ws: Union[Array, None],
    coarsening_factor: Union[int, None],
    observed_variables_sat: Union[List[str], None],
    observed_variables_ws: Union[List[str], None],
    observed_variables_coarsening: Union[List[str], None],
    sigma_y: Array,
    forcings: xarray.Dataset,
    target_template: xarray.Dataset,
    ckpt: gencast.CheckPoint,
    task_config: graphcast.TaskConfig,
    denoiser_config: denoiser.DenoiserArchitectureConfig,
    noise_encoder_config: denoiser.NoiseEncoderConfig,
    std_z: xarray.Dataset,
    min_x: xarray.Dataset,
    std_x: xarray.Dataset,
    mean_x: xarray.Dataset,
    noise_levels: Array,
    max_iter: int,
) -> Tuple[float, Array]:
    """
    Weighting step: compute normalized log pseudo-weights
    Input(s)
        - N (int): number of particles
        - N_thr_min (int): minimum number of efficient particles
        - N_thr_max (int): maximum number of efficient particles
        - alpha_init (float): first inflation coefficient
        - previous_particle_path (str): path to particles at time k
        - observation (Array): normalized observations hat{y}^{k+1} = [hat{y}^{k+1}_{ws}, hat{y}^{k+1}_{sat}] with dimension (batch=1, num_observed_variables)
        - mask_sat (Union[Array, None]): boolean Array of dimension (181, 360) corresponding to satellite observations
        - mask_ws (Union[Array, None]): boolean Array of dimension (181, 360) corresponding to ground observations
        - coarsening_factor (Union[int, None]): coarsening factor
        - observed_variables_sat (Union[List[str], None]): ordered list of variables observed by satellite
        - observed_variables_ws (Union[List[str], None]): ordered list of variables observed by ground weather stations
        - observed_variables_coarsening (Union[List[str], None]): ordered list of observed variables for coarsening
        - sigma_y (Array): covariance matrix of normalized observations with dimension (1, num_observed_variables)
        - forcings (xarray.Dataset): unnormalized forcing terms used by the GenCast denoiser
        - target_template (xarray.Dataset): template of the target with dimension (batch=1, time=1, lat=181, lon=360, levels=13)
        - ckpt (gencast.CheckPoint): checkpoint to use
        - task_config (graphcast.TaskConfig)
        - denoiser_config (denoiser.DenoiserArchitectureConfig)
        - noise_encoder_config (denoiser.NoiseEncoderConfig)
        - std_z (xarray.Dataset): standard deviations of residuals
        - min_x (xarray.Dataset): minimum values of unnnormalized states
        - std_x (xarray.Dataset): standard deviation of unnormalized states
        - mean_x (xarray.Dataset): mean of unnnormalized states
        - noise_levels (Array): array containing noise levels used during sampling
        - max_iter (int): maximum number of iterations to do when looking for a decent inflation coefficient
    Returns
        - alpha (float): inflation factor used to compute normalized log pseudo-weights
        - tilde_w (Array): normalized log pseudo-weights [log(tilde{w}^{k+1}_{(1)}), ..., log(tilde{w}^{k+1}_{(N)})] with dimension (N,)
    """

    @hk.transform_with_state
    def estimate_expectation(
        task_config: graphcast.TaskConfig,
        denoiser_architecture_config: denoiser.DenoiserArchitectureConfig,
        noise_encoder_config: denoiser.NoiseEncoderConfig,
        inputs: xarray.Dataset,
        target_template: xarray.Dataset,
        forcings: xarray.Dataset,
        std_z: xarray.Dataset,
        min_x: xarray.Dataset,
        std_x: xarray.Dataset,
        mean_x: xarray.Dataset,
        noise_levels: Array,
    ) -> xarray.Dataset:
        """
        Estimate E[x^{k+1} | hat{x}^{k}_{(i)}] in order to approximate p(hat{y}^{k+1} | x^{k}_{(i)})
        Input(s)
            - task_config (graphcast.TaskConfig)
            - denoiser_architecture_config (denoiser.DenoiserArchitectureConfig)
            - noise_encoder_config (denoiser.NoiseEncoderConfig)
            - inputs (xarray.Dataset): unnormalized previous state x^{k}_{(i)} of the system with dimension (batch=1, time=2, lat=181, lon=360, levels=13)
            - target_template (xarray.Dataset): template of the target with dimension (batch=1, time=1, lat=181, lon=360, levels=13)
            - forcings (xarray.Dataset): unnormalized forcing terms used by the GenCast denoiser
            - std_z (xarray.Dataset): standard deviations of residuals
            - min_x (xarray.Dataset): minimum values of unnnormalized states
            - std_x (xarray.Dataset): standard deviation of unnormalized states
            - mean_x (xarray.Dataset): mean of unnnormalized states
            - noise_levels (Array): array containing noise levels used during sampling
        Returns
            - estimation (xarray.Dataset): an estimation of E[x^{k+1} | x^{k}_{(i)}]
        """
        # 1) Instanciate a classical GenCast denoiser
        denoiser = GenCastDenoiser(
            task_config=task_config,
            denoiser_architecture_config=denoiser_architecture_config,
            noise_encoder_config=noise_encoder_config,
        )

        # 2) Clean the Sea Surface Temperature (SST) variable for inputs and forcings
        variable_to_clean = "sea_surface_temperature"
        if variable_to_clean in inputs.keys():
            clean_inputs = utils.clean_NaN(inputs, variable_to_clean, min_x[variable_to_clean])
        else:
            clean_inputs = inputs
        if variable_to_clean in forcings.keys():
            clean_forcings = utils.clean_NaN(forcings, variable_to_clean, min_x[variable_to_clean])
        else:
            clean_forcings = forcings

        # 3) Normalize inputs and forcings
        normalized_inputs = utils.normalize(clean_inputs, std_x, mean_x)
        normalized_forcings = utils.normalize(clean_forcings, std_x, mean_x)

        # 4) Instanciate hat_z_1
        noise_level = noise_levels[0]
        hat_z_1 = noise_level * samplers_utils.spherical_white_noise_like(target_template)

        # 5) Use the classical GenCast Denoiser to estimate E[hat{z}^{k+1} | x^{k}_{(i)}]
        bcast_noise = xarray_jax.DataArray(
            jnp.tile(noise_level, hat_z_1.sizes["batch"]), dims=("batch",)
        )
        hat_z = denoiser(
            inputs=normalized_inputs,
            noisy_targets=hat_z_1,
            noise_levels=bcast_noise,
            forcings=normalized_forcings,
        )

        # 6) Unnormalize residual and add the previous unnormalized state of the system
        estimation = utils.unnormalize_prediction_and_add_input(
            inputs=clean_inputs,
            norm_predictions=hat_z,
            std_z=std_z,
            std_x=std_x,
            mean_x=mean_x,
        )

        return estimation

    # Jitted version of the function
    estimate_expectation_jitted = jax.jit(
        lambda rng, i: estimate_expectation.apply(
            ckpt.params,
            {},
            rng,
            task_config=task_config,
            denoiser_architecture_config=denoiser_config,
            noise_encoder_config=noise_encoder_config,
            inputs=i,
            target_template=target_template,
            forcings=forcings,
            std_z=std_z,
            min_x=min_x,
            std_x=std_x,
            mean_x=mean_x,
            noise_levels=noise_levels,
        )[0]
    )

    # pmap version to run in parallel
    estimate_expectation_pmap = xarray_jax.pmap(estimate_expectation_jitted, dim="sample")

    def observation_operator(
        x: xarray.Dataset,
        mask_sat: Union[Array, None],
        mask_ws: Union[Array, None],
        coarsening_factor: Union[int, None],
        observed_variables_sat: Union[List[str], None],
        observed_variables_ws: Union[List[str], None],
        observed_variables_coarsening: Union[List[str], None],
        std_x: xarray.Dataset,
        mean_x: xarray.Dataset,
    ) -> Array:
        """
        Apply the observation operator H on a given input x.
        Input(s)
            - x (Array): input of the observation operator, an estimation of E[x^{k+1} | hat{z}^{k+1}_{t}, hat{x}^{k}] in the following
            - mask_sat (Union[Array, None]): boolean Array of dimension (181, 360) corresponding to satellite observations
            - mask_ws (Union[Array, None]): boolean Array of dimension (181, 360) corresponding to ground observations
            - coarsening_factor (Union[int, None]): coarsening factor
            - observed_variables_sat (Union[List[str], None]): ordered list of variables observed by satellite
            - observed_variables_ws (Union[List[str], None]): ordered list of variables observed by ground weather stations
            - observed_variables_coarsening (Union[List[str], None]): ordered list of observed variables for coarsening
            - std_x (xarray.Dataset): standard deviation of unnormalized states
            - mean_x (xarray.Dataset): mean of unnnormalized states
        Returns
            - observations (Array): H(x) as jnp.ndarray with dimensions (batch=1, num_observed_variables)
        """
        # Normalize the array
        x = utils.normalize(values=x, scales=std_x, locations=mean_x)

        # Extract coarsened observations
        if (coarsening_factor is not None) and (observed_variables_coarsening is not None):
            obs_coarsening = x[observed_variables_coarsening]
            obs_coarsening = utils.convert_xarray_to_jax(obs_coarsening)
            obs_coarsening = utils.coarsening(x=obs_coarsening, factor=coarsening_factor)
            obs_coarsening = obs_coarsening.reshape(obs_coarsening.shape[0], -1)
            observations = obs_coarsening

        else:
            # Extract observation from weather stations
            if (observed_variables_ws is not None) and (mask_ws is not None):
                obs_weather_stations = x[observed_variables_ws]
                obs_weather_stations = utils.convert_xarray_to_jax(obs_weather_stations)
                obs_weather_stations = obs_weather_stations[:, mask_ws, :]
                obs_weather_stations = obs_weather_stations.reshape((
                    obs_weather_stations.shape[0],
                    -1,
                ))
            else:
                obs_weather_stations = jnp.array([[]])

            # Extract observation from satellite
            if (observed_variables_sat is not None) and (mask_sat is not None):
                obs_satellite = x[observed_variables_sat]
                obs_satellite = utils.convert_xarray_to_jax(obs_satellite)
                obs_satellite = obs_satellite[:, mask_sat, :]
                obs_satellite = obs_satellite.reshape((
                    obs_satellite.shape[0],
                    -1,
                ))
            else:
                obs_satellite = jnp.array([[]])

            # Concatenate observations from ground stations and satellite
            observations = jnp.concatenate([obs_weather_stations, obs_satellite], axis=1)

        return observations

    def compute_unnormalized_pseudo_weights(
        observations: Array,
        mask_sat: Union[Array, None],
        mask_ws: Union[Array, None],
        coarsening_factor: Union[int, None],
        observed_variables_sat: Union[List[str], None],
        observed_variables_ws: Union[List[str], None],
        observed_variables_coarsening: Union[List[str], None],
        sigma_y: Array,
        std_x: xarray.Dataset,
        mean_x: xarray.Dataset,
        alpha: float,
        expectation: xarray.Dataset,
    ) -> float:
        """
        Compute the unnormalized log pseudo-weights given an estimation of E[x^{k+1} | x^{k}_{(i)}]
        These weights are referred to as “pseudo-weights” because the covariance matrix of normalized observations is modified (inflation)
        Input(s)
            - observation (Array): normalized observations hat{y}^{k+1} = [hat{y}^{k+1}_{ws}, hat{y}^{k+1}_{sat}] with dimension (batch=1, num_observed_variables)
            - mask_sat (Union[Array, None]): boolean Array of dimension (181, 360) corresponding to satellite observations
            - mask_ws (Union[Array, None]): boolean Array of dimension (181, 360) corresponding to ground observations
            - coarsening_factor (Union[int, None]): coarsening factor
            - observed_variables_sat (Union[List[str], None]): ordered list of variables observed by satellite
            - observed_variables_ws (Union[List[str], None]): ordered list of variables observed by ground weather stations
            - observed_variables_coarsening (Union[List[str], None]): ordered list of variables for coarsening
            - sigma_y (Array): covariance matrix of normalized observations with dimension (1, num_observed_variables)
            - std_x (xarray.Dataset): standard deviation of unnormalized states
            - mean_x (xarray.Dataset): mean of unnnormalized states
            - alpha (float): inflation coefficient
            - expectation (xarray.Dataset): an estimation of E[x^{k+1} | hat{x}^{k}_{(i)}] with dimension (batch=1, time=1, lat=181, lon=360, levels=13)
        Returns
            - hat_tilde_w (float): unnormalized log pseudo-weight for ancestor x^{k}_{(i)}
        """
        # 1) Apply the observation operator H to the expectation
        Hx = observation_operator(
            x=expectation,
            mask_sat=mask_sat,
            mask_ws=mask_ws,
            coarsening_factor=coarsening_factor,
            observed_variables_sat=observed_variables_sat,
            observed_variables_ws=observed_variables_ws,
            observed_variables_coarsening=observed_variables_coarsening,
            std_x=std_x,
            mean_x=mean_x,
        )

        # 2) Get the difference between the observation and H(E[x^{k+1} | hat{x}^{k}_{(i)}])
        v = observations - Hx

        # 3) Apply inflation to Sigma_{y}
        tilde_sigma_y = (1.0 / alpha) * sigma_y

        # 4) Compute the unnormalize pseudo-weight
        hat_tilde_w = jnp.sum((1.0 / tilde_sigma_y) * (v**2))
        hat_tilde_w = -0.5 * hat_tilde_w
        hat_tilde_w = hat_tilde_w.item()

        return hat_tilde_w

    def normalize_log_pseudo_weights(
        hat_tilde_w: Array,
    ) -> Array:
        """
        Normalize log pseudo-weights
        Input(s)
            - hat_tilde_w (Array): unnormalized log pseudo-weights with dimension (N,)
        Returns
            - tilde_w (Array): normalized log pseudo-weights with dimension (N,)
        """
        tilde_w = hat_tilde_w - jax.scipy.special.logsumexp(hat_tilde_w)
        return tilde_w

    def compute_Neff(log_pseudo_weights: Array) -> float:
        """
        Compute the number of efficient particles given an array containing normalized log pseudo-weights
        Input(s)
            - log_pseudo_weights (Array): normalized log pseudo-weights [log(tilde{w}^{k+1}_{(1)}), ..., log(tilde{w}^{k+1}_{(N)})] with dimension (N,)
        Returns
            - n_eff (float): number of effective particles
        """
        log_n_eff = -jax.scipy.special.logsumexp(2 * log_pseudo_weights)
        n_eff = jnp.exp(log_n_eff)
        n_eff = n_eff.item()
        return n_eff

    # 1) Estimate E[x^{k+1} | hat{x}^{k}_{(i)}] for each previous particle x^{k}_{(i)}
    print(" (Weighting)")
    num_gpus = len([device for device in jax.devices() if device.platform == "gpu"])
    assert int(N % num_gpus) == 0
    num_steps = int(N // num_gpus)

    # Loop on the number steps
    expectation_estimations = []
    print("     Computation of expectations estimation...")
    for i in tqdm(range(1, num_steps + 1)):
        # Get a batch of particles to do the job in parallel
        samples = []
        start_index = (i - 1) * num_gpus + 1
        for index in range(start_index, min(start_index + num_gpus, N + 1)):
            if previous_particle_path[-1] == "/":
                particle_path = previous_particle_path + str(index) + str(".nc")
            else:
                particle_path = previous_particle_path + str("/") + str(index) + str(".nc")
            with open(particle_path, "rb") as file:
                particle = xarray.load_dataset(file, decode_timedelta=True).compute()
            samples.append(particle)

        # Do computations in parallel
        key = jax.random.PRNGKey(np.random.randint(100_000))
        keys = jax.random.split(key, num_gpus)
        samples = xarray.concat(
            samples, dim=xarray.DataArray([j for j in range(len(samples))], dims="sample")
        )
        samples_expectations = estimate_expectation_pmap(keys, samples)

        # Update the list
        samples_expectations = [
            samples_expectations.isel(sample=j)
            for j in range(samples_expectations.sizes["sample"])
        ]
        expectation_estimations += samples_expectations

    # 2) Find the best inflation factor alpha
    print("     Looking for a decent inflation factor...")
    alpha_min, alpha_max, alpha = 1e-12, 1.0, alpha_init
    N_eff, num_iter = None, 0
    while num_iter < max_iter:
        # Compute unnormalized log pseudo-weights
        hat_tilde_w = []
        for i in range(len(expectation_estimations)):
            hat_tilde_w.append(
                compute_unnormalized_pseudo_weights(
                    observations=observations,
                    mask_sat=mask_sat,
                    mask_ws=mask_ws,
                    coarsening_factor=coarsening_factor,
                    observed_variables_sat=observed_variables_sat,
                    observed_variables_ws=observed_variables_ws,
                    observed_variables_coarsening=observed_variables_coarsening,
                    sigma_y=sigma_y,
                    std_x=std_x,
                    mean_x=mean_x,
                    alpha=alpha,
                    expectation=expectation_estimations[i],
                )
            )
        hat_tilde_w = jnp.asarray(hat_tilde_w)

        # Compute normalized log pseudo-weights
        tilde_w = normalize_log_pseudo_weights(hat_tilde_w=hat_tilde_w)

        # Compute the number of efficient particles
        N_eff = compute_Neff(log_pseudo_weights=tilde_w)
        print("         alpha={:.10f}, N_eff={:.4f}".format(alpha, N_eff))

        # Update alpha
        if N_eff > N_thr_max:
            alpha_min = alpha
            alpha = 0.5 * (alpha_min + alpha_max)
        elif N_eff < N_thr_min:
            alpha_max = alpha
            alpha = 0.5 * (alpha_min + alpha_max)
        else:
            break

        # Update the number of iterations
        num_iter += 1

    return (alpha, tilde_w)


def resampling(key: jax.random.PRNGKey, tilde_w: Array, method: str = "systematic"):
    """
    Resampling step: draw indices directly from Cat({w^{k+1}_{(i)}}) or using systematic resampling
    Input(s)
        - key (jax.random.PRNGKey): random key
        - tilde_w (Array): normalized log pseudo-weights [log(tilde{w}^{k+1}_{(1)}), ..., log(tilde{w}^{k+1}_{(N)})] with dimension (N,)
    Returns
        - indices (Array): new indices to use for the sampling step
    """
    print(" (Resampling)")
    N = tilde_w.shape[0]
    if method == "categorical":
        indices = jax.random.categorical(key, logits=tilde_w, shape=(N,))
    elif method == "systematic":
        weights = jnp.exp(tilde_w)
        weights = weights / jnp.sum(weights)
        cumulative_sum = jnp.cumsum(weights)
        u0 = jax.random.uniform(key, minval=0.0, maxval=1.0 / N)
        positions = u0 + jnp.arange(N) / N
        indices = jnp.searchsorted(cumulative_sum, positions, side="right")
    else:
        raise NotImplementedError(f"resampling method '{method}' hasn't been implemented!")
    return indices


def sampling(
    indices: Array,
    previous_particles_path: str,
    new_particles_path: str,
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
    observations: Array,
    mask_sat: Union[Array, None],
    mask_ws: Union[Array, None],
    coarsening_factor: Union[int, None],
    observed_variables_sat: Union[List[str], None],
    observed_variables_ws: Union[List[str], None],
    observed_variables_coarsening: Union[List[str], None],
    sigma_y: Array,
    solver: str,
    max_iter: int,
    tol: float,
):
    """
    Sampling step: draw samples from p(x^{k+1} | x^{k}_{a^{k+1}_{(i)}}, hat{y}^{k+1})
    Input(s)
        - indices (Array): indices [a^{k+1}_{(1)}, ..., a^{k+1}_{(N)}] to draw samples from with dimension (N,)
        - previous_particles_path (str): path of particles at time k
        - new_particles_path (str): path of particles at time k
        - target_template (xarray.Dataset): template with dimensions (batch=1, time=1, lat=181, lon=360, levels=13)
        - forcings (xarray.Dataset): forcings terms used by the GenCast denoiser with dimensions (batch=1, time=1, lat=181, lon=360, levels=13)
        - ckpt (gencast.CheckPoint): checkpoint to use
        - task_config (graphcast.TaskConfig)
        - denoiser_config (denoiser.DenoiserArchitectureConfig)
        - noise_encoder_config (denoiser.NoiseEncoderConfig)
        - sampler (str): sampler to use
        - sampler_config (Union[Any, gencast.SamplerConfig])
        - min_x (xarray.Dataset): minimum values of system states for each variable
        - std_x (xarray.Dataset): standard deviation of system states for each variable
        - std_z (xarray.Dataset): standard deviation of residuals for each variable
        - mean_x (xarray.Dataset): mean of system states for each variable
        - observations (Array): normalized observations hat{y}^{k+1} = [hat{y}^{k+1}_{ws}, hat{y}^{k+1}_{sat}] with dimension (batch=1, num_observed_variables)
        - mask_sat (Union[Array, None]): boolean Array of dimension (181, 360) corresponding to satellite observations
        - mask_ws (Union[Array, None]): boolean Array of dimension (181, 360) corresponding to ground observations
        - coarsening_factor (Union[int, None]): coarsening factor
        - observed_variables_sat (Union[List[str], None]): ordered list of variables observed by satellite
        - observed_variables_ws (Union[List[str], None]): ordered list of variables observed by ground weather stations
        - observed_variables_coarsening (Union[List[str], None]): ordered list of observed variables for coarsening
        - sigma_y (Array): covariance matrix of normalized observations with dimension (1, num_observed_variables)
        - solver (str): solver to use in MMPS iterations
        - max_iter (int): maximum number of iterations to do when solving the system in MMPS
        - tol (float): numerical tolerance used in the MMPS solver
    """

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
        observations: Array,
        mask_sat: Union[Array, None],
        mask_ws: Union[Array, None],
        coarsening_factor: Union[int, None],
        observed_variables_sat: Union[List[str], None],
        observed_variables_ws: Union[List[str], None],
        observed_variables_coarsening: Union[List[str], None],
        sigma_y: Array,
        solver: str,
        max_iter: int,
        tol: float,
    ) -> xarray.Dataset:
        """
        Draw a sample conditionally on an observation
        Input(s)
            - inputs (xarray.Dataset): previous states of the system with dimensions (batch=1, time=2, lat=181, lon=360, levels=13)
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
            - observations (Array): normalized observations hat{y}^{k+1} = [hat{y}^{k+1}_{ws}, hat{y}^{k+1}_{sat}] with dimension (batch=1, num_observed_variables)
            - mask_sat (Union[Array, None]): boolean Array of dimension (181, 360) corresponding to satellite observations
            - mask_ws (Union[Array, None]): boolean Array of dimension (181, 360) corresponding to ground observations
            - coarsening_factor (Union[int, None]): coarsening factor
            - observed_variables_sat (Union[List[str], None]): ordered list of variables observed by satellite
            - observed_variables_ws (Union[List[str], None]): ordered list of variables observed by ground weather stations
            - observed_variables_coarsening (Union[List[str], None]): ordered list of observed variables for coarsening
            - sigma_y (Array): covariance matrix of normalized observations with dimension (1, num_observed_variables)
            - solver (str): solver to use in MMPS iterations
            - max_iter (int): maximum number of iterations to do when solving the system in MMPS
            - tol (float): numerical tolerance used in the MMPS solver
        Returns
            - sample (xarray.Dataset): a sample drawn from p(x^{k+1} | x^{k}_{a^{k+1}_(i)}, hat{y}^{k+1})
        """
        # Instanciate a denoiser
        denoiser = ConditionalDenoiser(
            mask_satellite=mask_sat,
            mask_weather_stations=mask_ws,
            coarsening_factor=coarsening_factor,
            observed_variables_satellite=observed_variables_sat,
            observed_variables_weather_stations=observed_variables_ws,
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
    conditional_sampling_jitted = jax.jit(
        lambda rng, i: conditional_sampling.apply(
            ckpt.params,
            {},
            rng,
            inputs=i,
            target_template=target_template,
            forcings=forcings,
            task_config=task_config,
            denoiser_config=denoiser_config,
            noise_encoder_config=noise_encoder_config,
            sampler=sampler,
            sampler_config=sampler_config,
            min_x=min_x,
            std_x=std_x,
            std_z=std_z,
            mean_x=mean_x,
            observations=observations,
            mask_sat=mask_sat,
            mask_ws=mask_ws,
            coarsening_factor=coarsening_factor,
            observed_variables_sat=observed_variables_sat,
            observed_variables_ws=observed_variables_ws,
            observed_variables_coarsening=observed_variables_coarsening,
            sigma_y=sigma_y,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
        )[0]
    )

    # pmap version to run in parallel
    conditional_sampling_pmap = xarray_jax.pmap(conditional_sampling_jitted, dim="sample")

    # Draw a sample from p(x_{k}, x_{k-1}^{a_{k}^{(i)}}) for each i in indices
    print(" (Sampling)")
    N = indices.shape[0]
    num_gpus = len([device for device in jax.devices() if device.platform == "gpu"])
    assert int(N % num_gpus) == 0
    num_steps = int(N // num_gpus)

    # Loop on the number steps
    print("     Draw conditional samples...")
    count = 1
    for i in tqdm(range(1, num_steps + 1)):
        # Get a batch of particles to do the job in parallel
        samples = []
        start_index = (i - 1) * num_gpus + 1
        for index in range(start_index, min(start_index + num_gpus, N + 1)):
            if previous_particles_path[-1] == "/":
                particle_path = previous_particles_path + str(indices[index - 1] + 1) + str(".nc")
            else:
                particle_path = (
                    previous_particles_path + str("/") + str(indices[index - 1] + 1) + str(".nc")
                )
            with open(particle_path, "rb") as file:
                particle = xarray.load_dataset(file, decode_timedelta=True).compute()
            samples.append(particle)

        # Do computations in parallel
        key = jax.random.PRNGKey(np.random.randint(100_000))
        keys = jax.random.split(key, num_gpus)
        samples = xarray.concat(
            samples, dim=xarray.DataArray([j for j in range(len(samples))], dims="sample")
        )
        next_samples = conditional_sampling_pmap(keys, samples)

        # Convert to a list
        next_samples = [next_samples.isel(sample=j) for j in range(next_samples.sizes["sample"])]

        # Update the inputs for next step and save it
        for j, next_sample in enumerate(next_samples):
            next_input = xarray.merge([next_sample, forcings])
            next_input = next_input.drop_vars("total_precipitation_12hr")
            next_input = xarray.concat(
                [samples.isel(sample=j), next_input], dim="time", data_vars="minimal"
            )
            next_input = next_input.isel(time=slice(-2, None))
            if new_particles_path[-1] == "/":
                file_name = new_particles_path + str(count) + str(".nc")
            else:
                file_name = new_particles_path + str("/") + str(count) + str(".nc")
            next_input.to_netcdf(file_name, format="NETCDF4", engine="netcdf4")
            count += 1

        # Free memory
        del samples
        del next_samples
        gc.collect()
        jax.clear_caches()


def step(
    step_number: int,
    previous_particles_path: str,
    new_particles_path: str,
    N: int,
    N_thr_min: int,
    N_thr_max: int,
    alpha_init: float,
    observations: Array,
    mask_sat: Union[Array, None],
    mask_ws: Union[Array, None],
    coarsening_factor: Union[int, None],
    observed_variables_sat: Union[List[str], None],
    observed_variables_ws: Union[List[str], None],
    observed_variables_coarsening: Union[List[str], None],
    sigma_y: Array,
    forcings: xarray.Dataset,
    target_template: xarray.Dataset,
    ckpt: gencast.CheckPoint,
    task_config: graphcast.TaskConfig,
    denoiser_config: denoiser.DenoiserArchitectureConfig,
    noise_encoder_config: denoiser.NoiseEncoderConfig,
    sampler: str,
    sampler_config: Union[Dict, gencast.SamplerConfig],
    std_z: xarray.Dataset,
    min_x: xarray.Dataset,
    std_x: xarray.Dataset,
    mean_x: xarray.Dataset,
    noise_levels: Array,
    max_iter_alpha: int,
    solver: str,
    max_iter_solver: int,
    tol_solver: float,
):
    """
    - step_number (int): indice (k+1) of the time step
    - previous_particles_path (str): path of particles at time k
    - new_particles_path (str): path of particles at time (k+1)
    - N (int): number of particles
    - N_thr_min (int): minimum number of efficient particles
    - N_thr_max (int): maximum number of efficient particles
    - alpha_init (float): first inflation coefficient
    - observations (Array): normalized observations hat{y}^{k+1} = [hat{y}^{k+1}_{ws}, hat{y}^{k+1}_{sat}] with dimension (batch=1, num_observed_variables)
    - mask_sat (Union[Array, None]): boolean Array of dimension (181, 360) corresponding to satellite observations
    - mask_ws (Union[Array, None]): boolean Array of dimension (181, 360) corresponding to ground observations
    - coarsening_factor (Union[int, None]): coarsening factor
    - observed_variables_sat (Union[List[str], None]): ordered list of variables observed by satellite
    - observed_variables_ws (Union[List[str], None]): ordered list of variables observed by ground weather stations
    - observed_variables_coarsening (Union[List[str], None]): ordered list of observed variables for coarsening
    - sigma_y (Array): covariance matrix of normalized observations with dimension (1, num_observed_variables)
    - forcings (xarray.Dataset): unnormalized forcing terms used by the GenCast denoiser
    - target_template (xarray.Dataset): template of the target with dimension (batch=1, time=1, lat=181, lon=360, levels=13)
    - ckpt (gencast.CheckPoint): checkpoint to use
    - task_config (graphcast.TaskConfig)
    - denoiser_config (denoiser.DenoiserArchitectureConfig)
    - noise_encoder_config (denoiser.NoiseEncoderConfig)
    - sampler (str): sampler to use
    - sampler_config (Union[Any, gencast.SamplerConfig])
    - std_z (xarray.Dataset): standard deviations of residuals
    - min_x (xarray.Dataset): minimum values of unnnormalized states
    - std_x (xarray.Dataset): standard deviation of unnormalized states
    - mean_x (xarray.Dataset): mean of unnnormalized states
    - noise_levels (Array): array containing noise levels used during sampling
    - max_iter_alpha (int): maximum number of iterations to do when looking for a decent inflation factor
    - solver (str): solver to use in MMPS iterations
    - max_iter_solver (int): maximum number of iterations to do when solving the system in MMPS
    - tol_solver (float): numerical tolerance used in the MMPS solver
    """
    # Compute the weights and get the indices for sampling
    if step_number > 1:
        _, tilde_w = weighting(
            N=N,
            N_thr_min=N_thr_min,
            N_thr_max=N_thr_max,
            alpha_init=alpha_init,
            previous_particle_path=previous_particles_path,
            observations=observations,
            mask_sat=mask_sat,
            mask_ws=mask_ws,
            coarsening_factor=coarsening_factor,
            observed_variables_sat=observed_variables_sat,
            observed_variables_ws=observed_variables_ws,
            observed_variables_coarsening=observed_variables_coarsening,
            sigma_y=sigma_y,
            forcings=forcings,
            target_template=target_template,
            ckpt=ckpt,
            task_config=task_config,
            denoiser_config=denoiser_config,
            noise_encoder_config=noise_encoder_config,
            std_z=std_z,
            min_x=min_x,
            std_x=std_x,
            mean_x=mean_x,
            noise_levels=noise_levels,
            max_iter=max_iter_alpha,
        )
        indices = resampling(key=jax.random.PRNGKey(np.random.randint(100_000)), tilde_w=tilde_w)
    else:
        indices = jnp.asarray([i for i in range(N)])

    # Do sampling from the optimal proposal
    sampling(
        indices=indices,
        previous_particles_path=previous_particles_path,
        new_particles_path=new_particles_path,
        target_template=target_template,
        forcings=forcings,
        ckpt=ckpt,
        task_config=task_config,
        denoiser_config=denoiser_config,
        noise_encoder_config=noise_encoder_config,
        sampler=sampler,
        sampler_config=sampler_config,
        min_x=min_x,
        std_x=std_x,
        std_z=std_z,
        mean_x=mean_x,
        observations=observations,
        mask_sat=mask_sat,
        mask_ws=mask_ws,
        coarsening_factor=coarsening_factor,
        observed_variables_sat=observed_variables_sat,
        observed_variables_ws=observed_variables_ws,
        observed_variables_coarsening=observed_variables_coarsening,
        sigma_y=sigma_y,
        solver=solver,
        max_iter=max_iter_solver,
        tol=tol_solver,
    )


def filtering(
    data_path: str,
    output_path: str,
    checkpoint_path: str,
    N: int,
    N_thr_min: int,
    N_thr_max: int,
    alpha_init: float,
    mask_sat_path: Union[str, None],
    mask_ws_path: Union[str, None],
    coarsening_factor: Union[int, None],
    observed_variables_sat: Union[List[str], None],
    observed_variables_ws: Union[List[str], None],
    observed_variables_coarsening: Union[List[str], None],
    sigma_y_sat_path: Union[str, None],
    sigma_y_ws_path: Union[str, None],
    sigma_y_coarsening_path: Union[str, None],
    sampler: str,
    sampler_config: Union[Dict, gencast.SamplerConfig],
    std_z_path: str,
    min_x_path: str,
    std_x_path: str,
    mean_x_path: str,
    max_iter_alpha: int,
    solver: str,
    max_iter_solver: int,
    tol_solver: float,
):
    """
    Do filtering with the Fully-Adapted Auxiliary Particle Filter (FA-APF)
    Input(s)
        - data_path (str): path to the reference from which observations will be extracted
        - output_path (str): path to save the particles for each time step
        - checkpoint_path (str): path to the checkpoint (model) to use
        - N (int): number of particles
        - N_thr_min (int): minimum number of efficient particles
        - N_thr_max (int): maximum number of efficient particles
        - alpha_init (float): first inflation coefficient
        - mask_sat_path (Union[str, None]): path to satellite mask
        - mask_ws_path (Union[str, None]): path to ground weather stations mask
        - coarsening_factor (Union[int, None]): coarsening factor
        - observed_variables_sat (Union[List[str], None]): ordered list of variables observed by satellite
        - observed_variables_ws (Union[List[str], None]): ordered list of variables observed by ground weather stations
        - observed_variables_coarsening (Union[List[str], None]): ordered list of observed variables for coarsening
        - sigma_y_sat_path (Union[str, None]): path to the covariance matrix of unnormalized satellite observations (with dimension (len(observed_variables_sat), 13)))
        - sigma_y_ws_path (Union[str, None]): path to the covariance matrix of unnormalized ground observations (with dimension (len(observed_variables_ws),)
        - sigma_y_coarsening_path (Union[str, None]): path to the covariance matrix of unnormalized coarsened observations (with dimension (num_variables,)
        - sampler (str): sampler to use during the reverse diffusion process
        - sampler_config (Union[Dict, gencast.SamplerConfig]): configuration of the sampler
        - min_x_path (str): path to min_x statistic
        - std_x_path (str): path to std_x statistic
        - std_z_path (str): path to std_z statistic
        - mean_x_path (str): path to mean_x statistic
        - max_iter_alpha (int): maximum number of iterations to do when looking for a decent inflation factor
        - solver (str): solver to use in MMPS iterations
        - max_iter_solver (int): maximum number of iterations to do when solving the system in MMPS
        - tol_solver (float): numerical tolerance used in the MMPS solver
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
        noise_levels = samplers_utils.noise_schedule(
            max_noise_level=88.0,
            min_noise_level=2e-5,
            num_noise_levels=32,
            rho=6.0,
        )
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

    # Get the number of steps to do
    num_steps = targets.sizes["time"]
    assert num_steps == forcings.sizes["time"]

    # Duplicate initial conditions
    if output_path[-1] == "/":
        ic_folder = Path(output_path + str("0/"))
    else:
        ic_folder = Path(output_path + str("/0/"))
    ic_folder.mkdir(parents=True, exist_ok=True)
    for i in range(1, N + 1):
        if output_path[-1] == "/":
            file_name = output_path + str("0/") + str(i) + str(".nc")
        else:
            file_name = output_path + str("/0/") + str(i) + str(".nc")
        x0.to_netcdf(file_name, format="NETCDF4", engine="netcdf4")

    # Loop on the number of steps
    for i in range(1, num_steps + 1):
        # Define previous and new particles path
        print("Step n°{}".format(i))
        if output_path[-1] == "/":
            previous_particles_path = output_path + str(i - 1) + str("/")
            new_particles_path = output_path + str(i) + str("/")
        else:
            previous_particles_path = output_path + str("/") + str(i - 1) + str("/")
            new_particles_path = output_path + str("/") + str(i) + str("/")

        # Create the new particles folder
        new_folder = Path(new_particles_path)
        new_folder.mkdir(parents=True, exist_ok=True)

        # Count the number of files
        existing_files = list(new_folder.glob("*.nc"))
        if len(existing_files) >= N:
            print(
                f"→ {N} particles are already present for step {i}, moving directly to step {i + 1}!"
            )
        else:
            # Get current forcings and template
            current_forcings = forcings.isel(time=[i - 1])
            current_template = targets.isel(time=[i - 1])

            # Draw an observation
            current_observations = utils.draw_normalized_observations(
                x=current_template,
                std_x=std_x,
                mean_x=mean_x,
                min_x=min_x,
                sigma_y=sigma_hat_y,
                mask_satellite=mask_sat,
                mask_weather_stations=mask_ws,
                coarsening_factor=coarsening_factor,
                observed_variables_satellite=observed_variables_sat,
                observed_variables_weather_stations=observed_variables_ws,
                observed_variables_coarsening=observed_variables_coarsening,
            )

            # Save the current observation
            obs_file_name = new_particles_path + str("y.npy")
            np.save(obs_file_name, np.array(current_observations))

            # Apply the step function
            step(
                step_number=i,
                previous_particles_path=previous_particles_path,
                new_particles_path=new_particles_path,
                N=N,
                N_thr_min=N_thr_min,
                N_thr_max=N_thr_max,
                alpha_init=alpha_init,
                observations=current_observations,
                mask_sat=mask_sat,
                mask_ws=mask_ws,
                coarsening_factor=coarsening_factor,
                observed_variables_sat=observed_variables_sat,
                observed_variables_ws=observed_variables_ws,
                observed_variables_coarsening=observed_variables_coarsening,
                sigma_y=sigma_hat_y,
                forcings=current_forcings,
                target_template=current_template,
                ckpt=ckpt,
                task_config=ckpt.task_config,
                denoiser_config=denoiser_architecture_config,
                noise_encoder_config=ckpt.noise_encoder_config,
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

            # Free memory
            del current_forcings
            del current_template
            del current_observations
            gc.collect()
            jax.clear_caches()

        print("")

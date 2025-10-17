# Libraries
import gc
import haiku as hk
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import numpy as np
import warnings
import xarray

from jax import Array  # type: ignore
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Union

import sampler as Samplers
import utils

from denoisers import ConditionalDenoiser, GenCastDenoiser
from graphcast import denoiser, gencast, graphcast, samplers_utils, xarray_jax
from predictor import Predictor


def weighting(
    N: int,
    N_thr_min: int,
    N_thr_max: int,
    alpha_init: float,
    previous_particle_path: str,
    observations: Array,
    mask: Array,
    observed_variables: List[str],
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
        - observations (Array): normalized observations of the true state of the system at time k with dimension (batch=1, num_stations * len(self.observed_variables))
        - mask (Array): mask used to do subsampling with dimension (181, 360)
        - observed_variables (List[str]): ordered list of observed variables
        - sigma_y (Array): covariance matrix of normalized observations Sigma_{y} with dimension (len(observed_variables),)
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
        - max_iter (int): maximum number of iterations to do when looking for a decent inflation factor
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
        r"""
        Estimate E[x^{k+1} | hat{x}^{k}_{(i)}] in order to approximate p(hat{y}^{k+1} | x^{k}_{(i)})
        Input(s)
            - task_config (graphcast.TaskConfig)
            - denoiser_architecture_config (denoiser.DenoiserArchitectureConfig)
            - noise_encoder_config (denoiser.NoiseEncoderConfig)
            - inputs (xarray.Dataset): unnormalized previous states x^{k}_{(i)} of the system with dimension (batch=1, time=2, lat=181, lon=360, levels=13)
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

        # 7) Reintroduce NaNs in the prediction
        estimation = utils.reintroduce_nans(
            old_inputs=inputs, predictions=estimation, variable=variable_to_clean
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

    def compute_unnormalized_pseudo_weights(
        observations: Array,
        mask: Array,
        observed_variables: List[str],
        sigma_y: Array,
        std_x: xarray.Dataset,
        mean_x: xarray.Dataset,
        alpha: float,
        expectation: xarray.Dataset,
    ) -> float:
        r"""
        Compute the unnormalized log pseudo-weights given an estimation of E[x^{k+1} | x^{k}_{(i)}]
        These weights are referred to as “pseudo-weights” because the covariance matrix of normalized observations is modified (inflation)
        Input(s)
            - observation (Array): normalized observation at next time step (k+1) with dimension (batch=1, num_stations * len(self.observed_variables))
            - mask (Array): mask used to do subsampling with dimension (181, 360)
            - observed_variables (List[str]): ordered list of observed variables
            - sigma_y (Array): covariance matrix of normalized observations Sigma_{y} with dimension (len(observed_variables),)
            - std_x (xarray.Dataset): standard deviation of unnormalized states
            - mean_x (xarray.Dataset): mean of unnnormalized states
            - alpha (float): inflation factor
            - expectation (xarray.Dataset): an estimation of E[x^{k+1} | hat{x}^{k}_{(i)}] with dimension (batch=1, time=1, lat=181, lon=360, levels=13)
        Returns
            - hat_tilde_w (float): unnormalized log pseudo-weight for ancestor x^{k}_{(i)}
        """
        # 1) Apply the observation operator H to the expectation
        Hx = utils.normalize(values=expectation, scales=std_x, locations=mean_x)
        Hx = Hx[observed_variables]
        Hx = utils.convert_xarray_to_jax(Hx, False)
        Hx = jnp.array(Hx)
        Hx = Hx[:, mask, :]
        Hx = Hx.reshape((
            Hx.shape[0],
            -1,
        ))

        # 2) Get the difference between the observation and H(E[x^{k+1} | hat{x}^{k}_{(i)}])
        v = observations - Hx

        # 3) Apply inflation to Sigma_{y}
        tilde_sigma_y = (1.0 / alpha) * sigma_y
        num_stations = observations.shape[-1] // tilde_sigma_y.shape[-1]
        tilde_sigma_y = jnp.tile(tilde_sigma_y, num_stations).reshape((1, -1))

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

    def compute_Neff(log_pseudo_weights: Array):
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

    # Ignore warnings of GenCast (about sparsity)
    warnings.filterwarnings("ignore")

    # 1) Estimate E[x^{k+1} | hat{x}^{k}_{(i)}] for each previous particle x^{k}_{(i)}
    print(" (Weighting)")
    num_devices = len(jax.devices())
    if N % num_devices == 0:
        num_steps = N // num_devices
    else:
        num_steps = N // num_devices + 1

    # Loop on the number steps
    expectation_estimations = []
    print("     Computation of expectations estimation...")
    for i in tqdm(range(1, num_steps + 1)):
        samples = []
        start_index = (i - 1) * num_devices + 1

        # Get a batch of particles to do the job in parallel
        for index in range(start_index, min(start_index + num_devices, N + 1)):
            particle_path = previous_particle_path + str(index) + ".nc"
            with open(particle_path, "rb") as file:
                particle = xarray.load_dataset(file, decode_timedelta=True).compute()
            samples.append(particle)

        # Do computations in parallel
        key = jax.random.PRNGKey(np.random.randint(i * 1_000))
        keys = jax.random.split(key, num_devices)
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
    alpha_min, alpha_max, alpha = 1e-10, 1.0, alpha_init
    N_eff, num_iter = None, 0
    while num_iter < max_iter:
        # Compute unnormalized log pseudo-weights
        hat_tilde_w = []
        for i in range(len(expectation_estimations)):
            hat_tilde_w.append(
                compute_unnormalized_pseudo_weights(
                    observations=observations,
                    mask=mask,
                    observed_variables=observed_variables,
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
    Resampling step: draw indices from Cat({w^{k+1}_{(i)}})
    Input(s)
        - key (jax.random.PRNGKey): key used by the jax.random.categorical function
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
        raise NotImplementedError(f"resampling method '{method}' is not implemented!")
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
    mask: Array,
    observed_variables: List[str],
    sigma_y: Array,
    solver: str,
    max_iter: int,
    tol: float,
):
    """
    Sampling step: draw samples from p(x^{k+1} | x^{k}^{a_{k+1}_{(i)}}, \hat{y}^{k+1})
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
        - observations (Array): normalized observations of the true state of the system at time (k+1) with dimension (batch=1, num_stations * len(self.observed_variables))
        - mask (Array): mask used to do subsampling with dimension (181, 360)
        - observed_variables (List[str]): ordered list of observed variables
        - sigma_y (Array): covariance matrix of normalized observations Sigma_{y} with dimension (len(observed_variables),)
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
        mask: Array,
        observed_variables: List[str],
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
            - observations (Array): normalized observations of the true state of the system at time (k+1) with dimension (batch=1, num_stations * len(self.observed_variables))
            - mask (Array): mask used to do subsampling with dimension (181, 360)
            - observed_variables (List[str]): ordered list of observed variables
            - sigma_y (Array): covariance matrix of normalized observations Sigma_{y} with dimension (len(observed_variables),)
            - solver (str): solver to use in MMPS iterations
            - max_iter (int): maximum number of iterations to do when solving the system in MMPS
            - tol (float): numerical tolerance used in the MMPS solver
        Returns
            - sample (xarray.Dataset): a sample drawn from p(x^{k+1} | x^{k}^{(i)}, hat{y}^{k+1})
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
            mask=mask,
            observed_variables=observed_variables,
            sigma_y=sigma_y,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
        )[0]
    )

    # pmap version to run in parallel
    conditional_sampling_pmap = xarray_jax.pmap(conditional_sampling_jitted, dim="sample")

    # Ignore warnings of GenCast (about sparsity)
    warnings.filterwarnings("ignore")

    # Draw a sample from p(x_{k}, x_{k-1}^{a_{k}^{(i)}}) for each i in indices
    print(" (Sampling)")
    N = indices.shape[0]
    num_devices = len(jax.devices())
    if N % num_devices == 0:
        num_steps = N // num_devices
    else:
        num_steps = N // num_devices + 1

    # Loop on the number steps
    print("     Draw conditional samples...")
    count = 1
    for i in tqdm(range(1, num_steps + 1)):
        samples = []
        start_index = (i - 1) * num_devices + 1

        # Get a batch of particles to do the job in parallel
        for index in range(start_index, min(start_index + num_devices, N + 1)):
            particle_path = previous_particles_path + str(indices[index - 1] + 1) + ".nc"
            with open(particle_path, "rb") as file:
                particle = xarray.load_dataset(file, decode_timedelta=True).compute()
            samples.append(particle)

        # Do computations in parallel
        key = jax.random.PRNGKey(np.random.randint(i * 1000))
        keys = jax.random.split(key, num_devices)
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
            file_name = new_particles_path + str(count) + ".nc"
            next_input.to_netcdf(file_name)
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
    mask: Array,
    observed_variables: List[str],
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
    - previous_particle_path (str): path to particles at time k
    - observations (Array): normalized observations of the true state of the system at time (k+1) with dimension (batch=1, num_stations * len(self.observed_variables))
    - mask (Array): mask used to do subsampling with dimension (181, 360)
    - observed_variables (List[str]): ordered list of observed variables
    - sigma_y (Array): covariance matrix of normalized observations Sigma_{y} with dimension (len(observed_variables),)
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
    if step_number > 1:
        _, tilde_w = weighting(
            N=N,
            N_thr_min=N_thr_min,
            N_thr_max=N_thr_max,
            alpha_init=alpha_init,
            previous_particle_path=previous_particles_path,
            observations=observations,
            mask=mask,
            observed_variables=observed_variables,
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
        indices = resampling(
            key=jax.random.PRNGKey(np.random.randint(step_number * 1_000)), tilde_w=tilde_w
        )
    else:
        indices = jnp.asarray([i for i in range(N)])
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
        mask=mask,
        observed_variables=observed_variables,
        sigma_y=sigma_y,
        solver=solver,
        max_iter=max_iter_solver,
        tol=tol_solver,
    )


def filtering(
    filter_path: str,
    N: int,
    N_thr_min: int,
    N_thr_max: int,
    alpha_init: float,
    reference: xarray.Dataset,
    mask: Array,
    observed_variables: List[str],
    sigma_y: Array,
    x0: xarray.Dataset,
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
    r"""
    Do filtering with the Fully-Adapted Auxiliary Particle Filter (FA-APF)
    Input(s)
        - filter_path (str): path of the filter
        - N (int): number of particles
        - N_thr_min (int): minimum number of efficient particles
        - N_thr_max (int): maximum number of efficient particles
        - alpha_init (float): first inflation coefficient
        - reference (xarray.Dataset): reference trajectory from which observations are taken with dimension (batch=1, time=n, lat=181, lon=360, levels=13)
        - mask (Array): mask used to do subsampling with dimension (181, 360)
        - observed_variables (List[str]): ordered list of observed variables
        - sigma_y (Array): covariance matrix of normalized observations Sigma_{hat{y}} with dimension (len(observed_variables),)
        - x0 (xarray.Dataset): initial condition/first state of the system with dimension (batch=1, time=2, lat=181, lon=360, levels=13)
        - forcings (xarray.Dataset): unnormalized forcing terms used by the GenCast denoiser with dimension (batch=1, time=n, lat=181, lon=360, levels=13)
        - target_template (xarray.Dataset): template of the target with dimension (batch=1, time=n, lat=181, lon=360, levels=13)
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
    # Get the number of steps to do
    num_steps = target_template.sizes["time"]
    assert num_steps == forcings.sizes["time"]
    assert num_steps == reference.sizes["time"]

    # Duplicate initial conditions
    ic_folder = Path(filter_path + "/0/")
    ic_folder.mkdir(parents=True, exist_ok=True)
    for i in range(1, N + 1):
        file_name = filter_path + "/0/" + str(i) + ".nc"
        x0.to_netcdf(file_name)

    # Ignore warnings of GenCast (about sparsity)
    warnings.filterwarnings("ignore")

    # Loop on the number of steps
    for i in range(1, num_steps + 1):
        # Define previous and new particles path
        print("Step n°{}".format(i))
        previous_particles_path = filter_path + "/" + str(i - 1) + "/"
        new_particles_path = filter_path + "/" + str(i) + "/"

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

            # Extract observations
            current_observations = reference.isel(time=[i - 1])
            current_observations = utils.normalize(current_observations, std_x, mean_x)
            current_observations = current_observations[observed_variables]
            current_observations = utils.convert_xarray_to_jax(current_observations, False)
            current_observations = jnp.array(current_observations)
            current_observations = current_observations[:, mask, :]
            current_observations = current_observations.reshape((
                current_observations.shape[0],
                -1,
            ))

            # Add noise to the observation
            num_stations = jnp.count_nonzero(mask)
            std_y = jnp.sqrt(sigma_y)
            std_y = jnp.tile(sigma_y, (num_stations, 1))
            std_y = std_y[None, :, :]
            eps = np.random.randn(*std_y.shape)
            eps = std_y * eps
            eps = eps.reshape((
                eps.shape[0],
                -1,
            ))
            current_observations += eps

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
                mask=mask,
                observed_variables=observed_variables,
                sigma_y=sigma_y,
                forcings=current_forcings,
                target_template=current_template,
                ckpt=ckpt,
                task_config=task_config,
                denoiser_config=denoiser_config,
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

            # Free memory
            del current_forcings
            del current_template
            del current_observations
            gc.collect()
            jax.clear_caches()

        print("")

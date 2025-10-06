# Libraries
import haiku as hk
import jax.numpy as jnp  # type: ignore
import xarray

from abc import ABC, abstractmethod
from jax import Array  # type: ignore
from typing import Optional, Tuple, Union

import utils

from denoisers import ConditionalDenoiser, GenCastDenoiser
from graphcast import casting, gencast, samplers_utils, xarray_jax


class Sampler(ABC):
    """
    Base class for sampling.
    Child classes must implement:
        - __call__: to perform residual sampling given model inputs
    """

    def __init__(self, denoiser: Union[ConditionalDenoiser, GenCastDenoiser], **kwargs):
        """
        Initialize a sampler.
        Input(s)
            - denoiser (Union[ConditionalDenoiser, GenCastDenoiser]): the denoiser to use for the reverse diffusion process
        """
        # Denoiser attribute
        self._denoiser = denoiser

    def call_denoiser(
        self,
        noise_level: Array,
        inputs: xarray.Dataset,
        noisy_targets: xarray.Dataset,
        forcings: xarray.Dataset,
        observations: Optional[Array] = None,
    ) -> xarray.Dataset:
        """
        Use the pretrained denoiser to estimate E[hat{z}_{k} | hat{z}_{k}^{t}, hat{x}_{k-1}^{(i)}]
        or E[hat{z}_{k} | hat{z}_{k}^{t}, hat{x}_{k-1}^{(i)}, \hat{y}_{k}] if observations are used
        Input(s)
            - noise_level (Array): noise levels sigma_{t} in noisy targets
            - inputs (xarray.Dataset): normalized previous states hat{x}_{k-1}^{(i)} of the system with dimension (batch=1, time=2, lat=181, lon=360, levels=13)
            - noisy_targets (xarray.Dataset): noisy samples hat{z}_{k}^{t} at step t of the reverse diffusion process with dimensions (batch=1, time=1, lat=181, lon=360, levels=13)
            - forcings (xarray.Dataset): normalized forcing terms used by the GenCast denoiser
            - observation (Optional[Array]): normalized observations from fictional or real weather stations with dimension (batch=1, num_stations * num_observed_variables)
        Returns
            - output (xarray.Dataset): an estimation of E[hat{z}_{k} | hat{z}_{k}^{t}, hat{x}_{k-1}^{(i)}]
            or E[hat{z}_{k} | hat{z}_{k}^{t}, hat{x}_{k-1}^{(i)}, y_{k}] with dimensions (batch=1, time=1, lat=181, lon=360, levels=13)
        """
        bcast_noise = xarray_jax.DataArray(
            jnp.tile(noise_level, noisy_targets.sizes["batch"]), dims=("batch",)
        )
        if isinstance(self._denoiser, ConditionalDenoiser):
            return self._denoiser(
                observations=observations,
                inputs=inputs,
                noisy_targets=noisy_targets,
                noise_levels=bcast_noise,
                forcings=forcings,
            )
        else:
            return self._denoiser(
                inputs=inputs,
                noisy_targets=noisy_targets,
                noise_levels=bcast_noise,
                forcings=forcings,
            )

    @abstractmethod
    def __call__(
        self,
        inputs: xarray.Dataset,
        targets_template: xarray.Dataset,
        forcings: xarray.Dataset,
        observations: Optional[Array] = None,
        **kwargs,
    ) -> xarray.Dataset:
        """
        Sample residuals given the inputs/forcings and optionally an observation for conditional generation.
        Input(s)
            - inputs (xarray.Dataset): normalized previous states hat{x}_{k-1}^{(i)} with dimensions (batch=1, time=2, lat=181, lon=360, levels=13)
            - targets_template (xarray.Dataset): template of the target with dimension (batch=1, time=1, lat=181, lon=360, levels=13)
            - forcings (xarray.Dataset): normalized forcing terms used by the GenCast denoiser
            - observations (Optional[Array]): normalized observations from fictional or real weather stations with dimension (batch=1, num_stations * num_observed_variables)
        Returns
            - sample (xarray.Dataset): predicted residual (as xarray_jax) with dimensions (batch=1, time=1, lat=181, lon=360, levels=13)
        """
        ...


class DPM_Sampler(Sampler):
    """
    DPM sampler
    Input(s)
        - denoiser (Union[ConditionalDenoiser, GenCastDenoiser])
        - sampler_config (gencast.SamplerConfig)
    """

    def __init__(
        self,
        denoiser: Union[ConditionalDenoiser, GenCastDenoiser],
        sampler_config: gencast.SamplerConfig,
    ):
        # Denoiser attribute
        super().__init__(denoiser)

        # DPM sampler attributes
        self._noise_levels = samplers_utils.noise_schedule(
            sampler_config.max_noise_level,
            sampler_config.min_noise_level,
            sampler_config.num_noise_levels,
            sampler_config.rho,
        )
        self._stochastic_churn = sampler_config.stochastic_churn_rate > 0
        self._per_step_churn_rates = samplers_utils.stochastic_churn_rate_schedule(
            self._noise_levels,
            sampler_config.stochastic_churn_rate,
            sampler_config.churn_min_noise_level,
            sampler_config.churn_max_noise_level,
        )
        self._noise_level_inflation_factor = sampler_config.noise_level_inflation_factor

    def __call__(
        self,
        inputs: xarray.Dataset,
        targets_template: xarray.Dataset,
        forcings: xarray.Dataset,
        observations: Optional[Array] = None,
        **kwargs,
    ) -> xarray.Dataset:
        """
        Sample residuals using the two normalized previous states of the system and observations from weather stations
        Input(s)
            inputs (xarray.Dataset): normalized previous states hat{x}_{k-1}^{(i)} of the system with dimension (batch=1, time=2, lat=181, lon=360, levels=13)
            targets_template (xarray.Dataset): template of the target with dimension (batch=1, time=1, lat=181, lon=360, levels=13)
            forcings (xarray.Dataset): normalized forcing terms used by the GenCast denoiser
            observations (Optional[Array]): normalized observations from fictional or real weather stations with dimension (batch=1, num_stations * num_observed_variables)
        Returns
            sample (xaray.Dataset): predicted residual (as xarray_jax) with dimension (batch=1, time=1, lat=181, lon=360, levels=13)
        """
        # Define dtype, noise_levels and churn_rates
        dtype = casting.infer_floating_dtype(targets_template)
        noise_levels = jnp.array(self._noise_levels).astype(dtype)
        per_step_churn_rates = jnp.array(self._per_step_churn_rates).astype(dtype)

        # Partial function used in the body_fn
        def denoiser(noise_level: Array, hat_z_t: xarray.Dataset) -> xarray.Dataset:
            return self.call_denoiser(
                noise_level=noise_level,
                inputs=inputs,
                noisy_targets=hat_z_t,
                forcings=forcings,
                observations=observations,
            )

        # One step of the DPM sampler
        def body_fn(i: Array, hat_z_t: xarray.Dataset) -> xarray.Dataset:
            """
            One step of the DPM-Solver++ (https://arxiv.org/abs/2211.01095) sampling algorithm
            Input(s)
                i (Array): sampling iteration number
                hat_z_t (xarray.Dataset): noisy samples hat{z}_{k}^{t} at step t of the reverse diffusion process with dimensions (batch=1, time=1, lat=181, lon=360, levels=13)
            Returns
                hat_z_tp1 (xarray.Dataset): noisy_targets at iteration (i+1) with dimension (batch=1, time=1, lat=181, lon=360, levels=13)
            """

            # Function to generate noise on the sphere with sigma{t=1} as standard deviation
            def init_noise(template):
                return noise_levels[0] * samplers_utils.spherical_white_noise_like(template)

            # Add init_noise to the first residual
            maybe_init_noise = (i == 0).astype(noise_levels[0].dtype)
            hat_z_t = hat_z_t + init_noise(hat_z_t) * maybe_init_noise
            noise_level = noise_levels[i]

            # Apply noise inflation techniques
            if self._stochastic_churn:
                hat_z_t, noise_level = samplers_utils.apply_stochastic_churn(
                    x=hat_z_t,
                    noise_level=noise_level,
                    stochastic_churn_rate=per_step_churn_rates[i],
                    noise_level_inflation_factor=self._noise_level_inflation_factor,
                )

            # Compute residuals at the next step of the diffusion process
            next_noise_level = noise_levels[i + 1]
            mid_noise_level = jnp.sqrt(noise_level * next_noise_level)
            mid_over_current = mid_noise_level / noise_level

            hat_z = denoiser(noise_level, hat_z_t)
            z_mid = mid_over_current * hat_z_t + (1 - mid_over_current) * hat_z

            next_over_current = next_noise_level / noise_level
            hat_z_mid = denoiser(mid_noise_level, z_mid)
            hat_z_tp1 = next_over_current * hat_z_t + (1 - next_over_current) * hat_z_mid

            return samplers_utils.tree_where(next_noise_level == 0, hat_z, hat_z_tp1)

        # Loop on the body_fn function to solve the reverse diffusion equation
        noise_init = xarray.zeros_like(targets_template)
        return hk.fori_loop(0, len(noise_levels) - 1, body_fun=body_fn, init_val=noise_init)


class DDIM_Sampler(Sampler):
    """
    DDIM sampler
    Input(s)
        - denoiser (Union[ConditionalDenoiser, GenCastDenoiser])
        - noise_levels (Array): [sigma_{t=1}, ..., sigma_{t=0} = 0.] with dimension (num_steps + 1)
        - eta (float): the stochasticity parameter. If eta=1, it is equivalent to the DDPM sampler.
        - correction (bool): if True, correction steps are applied after the prediction step
        - num_correction_steps (Optional[int]): number of correction step to do
        - delta (Optional[float]): coefficient used in the correction step
    """

    def __init__(
        self,
        denoiser: Union[ConditionalDenoiser, GenCastDenoiser],
        noise_levels: Array,
        eta: float,
        correction: bool,
        num_correction_steps: Optional[int] = None,
        delta: Optional[float] = None,
    ):
        # Denoiser attribute
        super().__init__(denoiser)

        # DDIM attributes
        self._noise_levels = noise_levels
        self._eta = eta
        self.apply_correction = correction
        if correction and (num_correction_steps is None):
            self._num_corr_steps = 2
        else:
            self._num_corr_steps = num_correction_steps
        if correction and (delta is None):
            self._delta = 0.25
        else:
            self._delta = delta

    def __call__(
        self,
        inputs: xarray.Dataset,
        targets_template: xarray.Dataset,
        forcings: xarray.Dataset,
        observations: Optional[Array] = None,
        **kwargs,
    ) -> xarray.Dataset:
        """
        Sample residuals using the two normalized previous states of the system and observations from weather stations
        Input(s)
            inputs (xarray.Dataset): normalized previous states hat{x}_{k-1}^{(i)} of the system with dimension (batch=1, time=2, lat=181, lon=360, levels=13)
            targets_template (xarray.Dataset): template of the target with dimension (batch=1, time=1, lat=181, lon=360, levels=13)
            forcings (xarray.Dataset): normalized forcing terms used by the GenCast denoiser
            observations (Optional[Array]): normalized observations from fictional or real weather stations with dimension (batch=1, num_stations * num_observed_variables)
        Returns
            sample (xaray.Dataset): predicted residual (as xarray_jax) with dimension (batch=1, time=1, lat=181, lon=360, levels=13)
        """
        # Define dtype and noise_levels
        dtype = casting.infer_floating_dtype(targets_template)
        noise_levels = jnp.array(self._noise_levels).astype(dtype)

        # Partial function used in the body_fn
        def denoiser(noise_level: Array, hat_z_t: xarray.Dataset) -> xarray.Dataset:
            return self.call_denoiser(
                noise_level=noise_level,
                inputs=inputs,
                noisy_targets=hat_z_t,
                forcings=forcings,
                observations=observations,
            )

        # One step of the DDIM sampler
        def body_fn(i: Array, hat_z_t: xarray.Dataset) -> xarray.Dataset:
            """
            One step of the DDIM sampling algorithm (see https://azula.readthedocs.io/0.1.1/api/azula.sample.html#azula.sample.DDIMSampler)
            Input(s)
                i (Array): sampling iteration number
                hat_z_t (xarray.Dataset): noisy samples hat{z}_{k}^{t} at step t of the reverse diffusion process with dimensions (batch=1, time=1, lat=181, lon=360, levels=13)
            Returns
                hat_z_tp1 (xarray.Dataset): noisy_targets at iteration (i+1) with dimension (batch=1, time=1, lat=181, lon=360, levels=13)
            """

            # Function to generate noise on the sphere with sigma{t=1} as standard deviation
            def init_noise(template):
                return noise_levels[0] * samplers_utils.spherical_white_noise_like(template)

            # Add init_noise to the first residual
            maybe_init_noise = (i == 0).astype(noise_levels[0].dtype)
            hat_z_t = hat_z_t + init_noise(hat_z_t) * maybe_init_noise

            # Prediction step: compute residuals at the next step of the diffusion process
            sigma_t = noise_levels[i]
            sigma_s = noise_levels[i + 1]
            sigma_s_safe = sigma_s + 1e-4

            tau = 1.0 - ((sigma_s / sigma_t) ** 2)
            tau = jnp.clip(self._eta * tau, 0.0, 1.0)
            eps = samplers_utils.spherical_white_noise_like(hat_z_t)

            hat_z = denoiser(sigma_t, hat_z_t)

            hat_z_tp1 = hat_z
            hat_z_tp1 = hat_z_tp1 + sigma_s * (jnp.sqrt(1.0 - tau) / sigma_t) * (hat_z_t - hat_z)
            hat_z_tp1 = hat_z_tp1 + sigma_s * jnp.sqrt(tau) * eps

            # Correction step
            if self.apply_correction:
                for _ in range(self._num_corr_steps):
                    eps = samplers_utils.spherical_white_noise_like(hat_z_tp1)
                    s = (denoiser(sigma_s_safe, hat_z_tp1) - hat_z_tp1) / (sigma_s_safe**2)
                    gamma = self._delta * (sigma_s_safe**2)
                    hat_z_tp1 = hat_z_tp1 + 0.5 * gamma * s + jnp.sqrt(gamma) * eps

            return samplers_utils.tree_where(sigma_s == 0, hat_z, hat_z_tp1)

        # Loop on the body_fn function to solve the reverse diffusion equation
        noise_init = xarray.zeros_like(targets_template)
        return hk.fori_loop(0, len(noise_levels) - 1, body_fun=body_fn, init_val=noise_init)


class ABSampler(Sampler):
    @staticmethod
    def adams_bashforth_coeffs(rho: Array, n: int) -> Array:
        """
        Returns the coefficients of the :math:`n`-th order Adams-Bashforth method
        Input(s):
            - rho (Array): the integration variable
            - n (int): the order of the method
        Returns:
            - coeffs (Array): coefficients of the method
        """
        m = rho.shape[0] - 1
        n = min(n, m)
        k = jnp.arange(n, dtype=jnp.float64)
        V = rho[m - n : m] ** k[:, None]
        b = rho[m] ** (k + 1) / (k + 1) - rho[m - 1] ** (k + 1) / (k + 1)
        coeffs = jnp.linalg.solve(V, b).astype(rho.dtype)
        return coeffs

    def __init__(
        self,
        denoiser: Union[ConditionalDenoiser, GenCastDenoiser],
        noise_levels: Array,
        order: int,
        correction: bool,
        num_correction_steps: Optional[int] = None,
        delta: Optional[float] = None,
    ):
        """
        Adams-Bashforth multi-step sampler (aka LMS sampler).
        Input(s):
            - denoiser (Union[ConditionalDenoiser, GenCastDenoiser])
            - noise_levels (Array): [sigma_{t=1}, ..., sigma_{t=0} = 0.] with dimension (num_steps + 1)
            - order (int): order of the Adam-Bashforth solver
            - correction (bool): if True, correction steps are applied after the prediction step
            - num_correction_steps (Optional[int]): number of correction step to do
            - delta (Optional[float]): coefficient used in the correction step
        """
        # Denoiser attribute
        super().__init__(denoiser)

        # ABS attributes
        self._noise_levels = noise_levels
        self._order = order
        self.apply_correction = correction
        if correction and (num_correction_steps is None):
            self._num_corr_steps = 2
        else:
            self._num_corr_steps = num_correction_steps
        if correction and (delta is None):
            self._delta = 0.25
        else:
            self._delta = delta
        self.coefficients = []
        for i in range(len(noise_levels) - 1):
            if i < (self._order - 1):
                zeros = jnp.zeros(self._order - (i + 1))
                coeffs = self.adams_bashforth_coeffs(self._noise_levels[: i + 2], self._order)
                self.coefficients.append(jnp.concat((zeros, coeffs), axis=0))
            else:
                self.coefficients.append(
                    self.adams_bashforth_coeffs(self._noise_levels[: i + 2], self._order)
                )
        self.coefficients = jnp.stack(self.coefficients, axis=0)

    def __call__(
        self,
        inputs: xarray.Dataset,
        targets_template: xarray.Dataset,
        forcings: xarray.Dataset,
        observations: Optional[Array] = None,
        **kwargs,
    ) -> xarray.Dataset:
        """
        Sample residuals using the two normalized previous states of the system and observations from weather stations
        Input(s)
            inputs (xarray.Dataset): normalized previous states hat{x}_{k-1}^{(i)} of the system with dimension (batch=1, time=2, lat=181, lon=360, levels=13)
            targets_template (xarray.Dataset): template of the target with dimension (batch=1, time=1, lat=181, lon=360, levels=13)
            forcings (xarray.Dataset): normalized forcing terms used by the GenCast denoiser
            observations (Optional[Array]): normalized observations from fictional or real weather stations with dimension (batch=1, num_stations * num_observed_variables)
        Returns
            sample (xaray.Dataset): predicted residual (as xarray_jax) with dimension (batch=1, time=1, lat=181, lon=360, levels=13)
        """
        # Define dtype and noise_levels
        dtype = casting.infer_floating_dtype(targets_template)
        noise_levels = jnp.array(self._noise_levels).astype(dtype)

        # Partial function used in the body_fn
        def denoiser(noise_level: Array, hat_z_t: xarray.Dataset) -> xarray.Dataset:
            return self.call_denoiser(
                noise_level=noise_level,
                inputs=inputs,
                noisy_targets=hat_z_t,
                forcings=forcings,
                observations=observations,
            )

        def body_fn(
            i: Array, buf_and_state: Tuple[xarray.Dataset, xarray.Dataset]
        ) -> Tuple[xarray.Dataset, xarray.Dataset]:
            """
            One step of the Adam-Bashforth method (see https://azula.readthedocs.io/stable/api/azula.sample.html#azula.sample.ABSampler)
            Input(s)
                i (Array): sampling iteration number
                buf_and_state (Tuple[xarray.Dataset, xarray.Dataset]): buffer and current residual
            """
            # Get the buffer and current residual
            buf, hat_z_t = buf_and_state

            # Function to generate noise on the sphere with sigma{t=1} as standard deviation
            def init_noise(template):
                return noise_levels[0] * samplers_utils.spherical_white_noise_like(template)

            # Add init_noise to the first residual
            maybe_init_noise = (i == 0).astype(noise_levels[0].dtype)
            hat_z_t = hat_z_t + init_noise(hat_z_t) * maybe_init_noise

            # Use the denoiser to estimate hat_z
            sigma_t = noise_levels[i]
            hat_z = denoiser(sigma_t, hat_z_t)
            z_t = (hat_z_t - hat_z) / sigma_t

            # Update the buffer
            buf = xarray.concat([buf, z_t], dim="batch")
            if buf.sizes["batch"] > self._order:
                buf = buf.isel(batch=slice(-self._order, None))

            # Get coefficients
            coeffs = self.coefficients[i]

            # Combine past residual to get the next one
            buf_jnp = utils.convert_xarray_to_jax(buf, jax_array=False)
            coeffs = coeffs[:, None, None, None]
            weighted = coeffs * buf_jnp
            integral = weighted.sum(axis=0, keepdims=True)
            integral = utils.convert_jax_to_xarray(integral, template_dataset=z_t)
            hat_z_tp1 = hat_z_t + integral

            # Correction step
            sigma_s = noise_levels[i + 1]
            sigma_s_safe = sigma_s + 1e-4
            hat_z_tp1_corrected = hat_z_tp1
            if self.apply_correction:
                for _ in range(self._num_corr_steps):
                    eps = samplers_utils.spherical_white_noise_like(hat_z_tp1_corrected)
                    s = (denoiser(sigma_s_safe, hat_z_tp1_corrected) - hat_z_tp1_corrected) / (
                        sigma_s_safe**2
                    )
                    gamma = self._delta * (sigma_s_safe**2)
                    hat_z_tp1_corrected = (
                        hat_z_tp1_corrected + 0.5 * gamma * s + jnp.sqrt(gamma) * eps
                    )

            return samplers_utils.tree_where(
                sigma_s == 0, (buf, hat_z_tp1), (buf, hat_z_tp1_corrected)
            )

        # Loop on the body_fn function to solve the reverse diffusion equation
        buffer_init = [xarray.zeros_like(targets_template) for _ in range(self._order)]
        buffer_init = xarray.concat(buffer_init, dim="batch")
        noise_init = xarray.zeros_like(targets_template)
        _, hat_z = hk.fori_loop(0, len(noise_levels) - 1, body_fn, (buffer_init, noise_init))
        return hat_z

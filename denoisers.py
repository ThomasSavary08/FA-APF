# Libraries
import jax  # type: ignore
import xarray

from jax import Array  # type: ignore
from jax import numpy as jnp  # type: ignore
from typing import List, Optional

import linalg
import utils

from graphcast import denoiser, graphcast, xarray_jax


class GenCastDenoiser:
    """
    Classical GenCast denoiser returning E[hat{z}^{k+1} | hat{z}^{k+1}_{t}, hat{x}^{k}].
    Input(s)
        - task_config (graphcast.TaskConfig)
        - denoiser_architecture_config (denoiser.DenoiserArchitectureConfig)
        - noise_encoder_config (denoiser.NoiseEncoderConfig)
    """

    def __init__(
        self,
        task_config: graphcast.TaskConfig,
        denoiser_architecture_config: denoiser.DenoiserArchitectureConfig,
        noise_encoder_config: Optional[denoiser.NoiseEncoderConfig] = None,
    ):
        # Get the number of output variables
        num_surface_vars = len(
            set(task_config.target_variables) - set(graphcast.ALL_ATMOSPHERIC_VARS)
        )
        num_atmospheric_vars = len(
            set(task_config.target_variables) & set(graphcast.ALL_ATMOSPHERIC_VARS)
        )
        num_outputs = num_surface_vars + len(task_config.pressure_levels) * num_atmospheric_vars

        # Instanciate a denoiser
        denoiser_architecture_config.node_output_size = num_outputs
        self._denoiser = denoiser.Denoiser(noise_encoder_config, denoiser_architecture_config)

    def _c_in(self, noise_scale: xarray.DataArray) -> xarray.DataArray:
        """
        Compute the input scaling coefficient from EDM paper
        Input(s)
            - noise_scale (xarray.DataArray)
        Returns
            - c_in (xarray.DataArray)
        """
        c_in = (noise_scale**2 + 1) ** (-0.5)
        return c_in

    def _c_out(self, noise_scale: xarray.DataArray) -> xarray.DataArray:
        """
        Compute the output scaling coefficient from EDM paper
        Input(s)
            - noise_scale (xarray.DataArray)
        Returns
            - c_out (xarray.DataArray)
        """
        c_out = noise_scale * (noise_scale**2 + 1) ** (-0.5)
        return c_out

    def _c_skip(self, noise_scale: xarray.DataArray) -> xarray.DataArray:
        """
        Compute the skip scaling coefficient from EDM paper
        Input(s)
            - noise_scale (xarray.DataArray)
        Returns
            - c_skip (xarray.DataArray)
        """
        c_skip = 1 / (noise_scale**2 + 1)
        return c_skip

    def __call__(
        self,
        inputs: xarray.Dataset,
        noisy_targets: xarray.Dataset,
        noise_levels: xarray.DataArray,
        forcings: Optional[xarray.Dataset] = None,
        **kwargs,
    ) -> xarray.Dataset:
        """
        Call the denoiser to estimate E[hat{z}^{k+1} | hat{z}^{k+1}_{t}, hat{x}^{k}].
        Input(s)
            - inputs (xarray.Dataset): normalized previous states hat{x}^{k} of the system with dimension (batch=1, time=2, lat=181, lon=360, levels=13)
            - noisy_targets (xarray.Dataset): noisy samples hat{z}^{k+1}_{t} at step t of the reverse diffusion process with dimension (batch=1, time=1, lat=181, lon=360, levels=13)
            - noise_levels (xarray.Dataset): noise levels sigma_{t} in noisy targets such that Sigma_{t} = sigma_{t}^{2} * I
            - forcings (xarray.Dataset): normalized forcing terms used by the GenCast denoiser
        Returns
            - output (xarray.Dataset): an estimation of E[hat{z}^{k+1} | hat{z}^{k+1}_{t}, hat{x}^{k}] with dimensions (batch=1, time=1, lat=181, lon=360, levels=13)
        """
        # Compute F_{theta}(c_{in}x; c_{noise}(sigma)), the see EDM paper
        raw_predictions = self._denoiser(
            inputs=inputs,
            noisy_targets=noisy_targets * self._c_in(noise_levels),
            noise_levels=noise_levels,
            forcings=forcings,
            **kwargs,
        )

        # Compute c_{skip}(sigma)x + c_{out}(sigma)F_{theta}(c_{in}x; c_{noise}(sigma)), see the EDM paper
        output = raw_predictions * self._c_out(noise_levels) + noisy_targets * self._c_skip(
            noise_levels
        )
        return output


class ConditionalDenoiser:
    """
    Modified GenCast denoiser to estimate E[hat{z}^{k+1} | hat{z}^{k+1}_{t}, hat{x}^{k}, \hat{y}^{k+1}] with MMPS.
    See "Learning Diffusion Priors from Observations by Expectation Maximization" from Rozet et al for more details.
    Input(s)
        - mask (Array): mask used to do subsampling with dimension (181, 360)
        - observed_variables (List[str]): ordered list of observed variables
        - sigma_y (Array): covariance matrix of normalized observations Sigma_{y} with dimension (len(observed_variables),)
        - std_z (xarray.Dataset): standard deviations of residuals
        - std_x (xarray.Dataset): standard deviations of system states
        - mean_x (xarray.Dataset): means of system states
        - task_config (graphcast.TaskConfig)
        - denoiser_architecture_config (denoiser.DenoiserArchitectureConfig)
        - noise_encoder_config (denoiser.NoiseEncoderConfig)
        - solver (str): solver to use in MMPS iterations
        - max_iter (int): maximum number of iterations to do when solving the system in MMPS
        - tol (float): numerical tolerance used in the MMPS solver
    """

    def __init__(
        self,
        mask: Array,
        observed_variables: List[str],
        sigma_y: Array,
        std_z: xarray.Dataset,
        std_x: xarray.Dataset,
        mean_x: xarray.Dataset,
        task_config: graphcast.TaskConfig,
        denoiser_architecture_config: denoiser.DenoiserArchitectureConfig,
        noise_encoder_config: Optional[denoiser.NoiseEncoderConfig] = None,
        solver: Optional[str] = "bicgstab",
        max_iter: Optional[int] = 2,
        tol: Optional[float] = 1e-8,
    ):
        # Statistics attributes
        self.std_z = std_z
        self.std_x = std_x
        self.mean_x = mean_x

        # Observations attributes
        self.mask = mask
        self.observed_variables = observed_variables
        self.sigma_y = sigma_y

        # Solver attributes
        if solver == "cg":
            self.solver = jax.scipy.sparse.linalg.bicgstab
        elif solver == "bicgstab":
            self.solver = jax.scipy.sparse.linalg.bicgstab
        else:
            self.solver = jax.scipy.sparse.linalg.gmres
        self.max_iter = max_iter
        self.r_tol = tol

        # Classical GenCast denoiser
        self._denoiser = GenCastDenoiser(
            task_config, denoiser_architecture_config, noise_encoder_config
        )

    def denoise_and_predict(
        self,
        noisy_targets: Array,
        targets_template: xarray.Dataset,
        inputs: xarray.Dataset,
        noise_levels: xarray.DataArray,
        forcings: Optional[xarray.Dataset] = None,
        **kwargs,
    ) -> Array:
        """
        Call the classical GenCast denoiser and then compute E[\hat{x}^{k+1} | hat{z}^{k+1}_{t}, hat{x}^{k}]
        Input(s)
            - noisy_targets (Array): noisy samples hat{z}^{k+1}_{t} at step t of the reverse diffusion process as jnp.ndarray
            - targets_template (xarray.Dataset): template used by the convert_jax_to_xarray function
            - inputs (xarray.Dataset): normalized previous states hat{x}^{k} of the system with dimension (batch=1, time=2, lat=181, lon=360, levels=13)
            - noise_levels (xarray.Dataset): noise levels sigma_{t} in noisy targets such that Sigma_{t} = sigma_{t}^{2} * I
            - forcings (xarray.Dataset): normalized forcing terms used by the GenCast denoiser
        Returns
            - output (Array): an estimation of E[x^{k+1} | hat{z}^{k+1}_{t}, hat{x}^{k}] as jnp.ndarray
        """
        # Convert the array from jax to xarray_jax
        noisy_targets_xarray = utils.convert_jax_to_xarray(noisy_targets, targets_template)

        # Call the classical GenCast denoiser to estimate E[hat{z}^{k+1} | hat{z}^{k+1}_{t}, hat{x}^{k}]
        hat_z = self._denoiser(
            inputs=inputs,
            noisy_targets=noisy_targets_xarray,
            noise_levels=noise_levels,
            forcings=forcings,
            **kwargs,
        )

        # Get unnormalized inputs
        unnormalized_inputs = utils.unnormalize(inputs, self.std_x, self.mean_x)

        # Compute E[x^{k+1} | hat{z}^{k+1}_{t}, hat{x}^{k}] using E[hat{z}^{k+1} | hat{z}^{k+1}_{t}, hat{x}^{k}]
        x = utils.unnormalize_prediction_and_add_input(
            inputs=unnormalized_inputs,
            norm_predictions=hat_z,
            std_z=self.std_z,
            std_x=self.std_x,
            mean_x=self.mean_x,
        )

        # Compute E[\hat{x}^{k+1} | hat{z}^{k+1}_{t}, hat{x}^{k}] using E[hat{z}^{k+1} | hat{z}^{k+1}_{t}, hat{x}^{k}]
        x = utils.normalize(values=x, scales=self.std_x, locations=self.mean_x)

        # Convert the result back to jax
        output = utils.convert_xarray_to_jax(x)
        return output

    def observation_operator(
        self,
        x: Array,
        target_template: xarray.Dataset,
    ) -> Array:
        """
        Apply the observation operator H on a given input x.
        Input(s)
            - x (Array): input of the observation operator, an estimation of E[\hat{x}^{k+1} | hat{z}^{k+1}_{t}, hat{x}^{k}] in the following
            - target_templates (xarray.Dataset): template used by utils.convert_jax_to_xarray
        Returns
            - output (Array): H(x) as jnp.ndarray with dimensions (batch=1, num_stations * len(self.observed_variables))
        """
        # Convert the x to an xarray
        x = utils.convert_jax_to_xarray(x, target_template)

        # Extract observed variables
        output = x[self.observed_variables]
        output = utils.convert_xarray_to_jax(output)
        output = output[:, self.mask, :]
        output = output.reshape((
            output.shape[0],
            -1,
        ))

        return output

    def __call__(
        self,
        observations: Array,
        inputs: xarray.Dataset,
        noisy_targets: xarray.Dataset,
        noise_levels: xarray.DataArray,
        forcings: Optional[xarray.Dataset] = None,
        **kwargs,
    ) -> xarray.Dataset:
        """
        Call the conditional denoiser to estimate E[hat{z}^{k+1} | hat{z}^{k+1}_{t}, hat{x}^{k}, \hat{y}^{k+1}].
        Input(s)
            - observation (Array): normalized observations \hat{y}^{k+1} with dimension (batch=1, num_stations * len(self.observed_variables))
            - inputs (xarray.Dataset): normalized previous states of the system hat{x}^{k} with dimensions (batch=1, time=2, lat=181, lon=360, levels=13)
            - noisy_targets (xarray.Dataset): noisy samples hat{z}^{k+1}_{t} at step t of the reverse diffusion process with dimensions (batch=1, time=1, lat=181, lon=360, levels=13)
            - noise_levels (xarray.Dataset): noise levels sigma_{t} in noisy targets such that Sigma_{t} = sigma_{t}^{2} * I
            - forcings (xarray.Dataset): normalized forcing terms used by the GenCast denoiser
        Returns
            - output (xarray.Dataset): an estimation of E[hat{z}^{k+1} | hat{z}^{k+1}_{t}, hat{x}^{k}, \hat{y}^{k+1}] with dimensions (batch=1, time=1, lat=181, lon=360, levels=13)
        """
        # Convert hat{z}^{k+1}_{t} from xarray_jax to jax
        hat_z_kp1_t = utils.convert_xarray_to_jax(noisy_targets)
        hat_z_kp1_t = jnp.array(hat_z_kp1_t)

        # Use the pretrained GenCast denoiser to estimate E[\hat{x}^{k+1} | hat{z}^{k+1}_{t}, hat{x}^{k}]
        hat_x_kp1, vjp = jax.vjp(
            lambda current_residual: self.denoise_and_predict(
                noisy_targets=current_residual,
                targets_template=noisy_targets,
                inputs=inputs,
                noise_levels=noise_levels,
                forcings=forcings,
                **kwargs,
            ),
            hat_z_kp1_t,
        )

        # Linearize the observation operator H
        y, H = jax.linearize(
            lambda expectancy_estimation: self.observation_operator(
                x=expectancy_estimation, target_template=noisy_targets
            ),
            hat_x_kp1,
        )
        Ht = linalg.transpose(H, hat_x_kp1)

        # Compute the score of p(\hat{y}^{k+1}| hat{z}^{k+1}_{t},  hat{x}^{k})
        sigma_t = jnp.array(xarray_jax.unwrap_data(noise_levels))[..., None] ** 2
        num_stations = observations.shape[-1] // self.sigma_y.shape[-1]
        sigma_y_reshaped = jnp.tile(self.sigma_y, num_stations).reshape((1, -1))
        Ax = lambda v: sigma_y_reshaped * v + sigma_t * H(*vjp(Ht(v)))
        b = observations - y
        v, _ = self.solver(A=Ax, b=b, tol=self.r_tol, maxiter=self.max_iter)
        (score,) = vjp(Ht(v))
        score = utils.convert_jax_to_xarray(score, noisy_targets)

        # Get unnormalized inputs
        unnormalized_inputs = utils.unnormalize(inputs, self.std_x, self.mean_x)

        # Compute E[hat{z}^{k+1} | hat{z}^{k+1}_{t}, hat{x}^{k}]
        hat_x_kp1 = utils.convert_jax_to_xarray(hat_x_kp1, noisy_targets)
        x_kp1 = utils.unnormalize(hat_x_kp1, self.std_x, self.mean_x)
        hat_z_kp1 = utils.substract_input_and_normalize_target(
            inputs=unnormalized_inputs,
            targets=x_kp1,
            std_z=self.std_z,
            std_x=self.std_x,
            mean_x=self.mean_x,
        )

        # Compute E[hat{z}^{k+1} | hat{z}^{k+1}_{t}, hat{x}^{k}, \hat{y}^{k+1}]
        output = hat_z_kp1 + sigma_t * score
        return output

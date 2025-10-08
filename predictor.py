# Libraries
import xarray

from jax import Array  # type: ignore
from typing import Optional

import utils

from sampler import Sampler


class Predictor:
    """
    High-level class to draw a sample from p(x^{k+1} | x^{k}) or p(x^{k+1} | x^{k}, y^{k+1})
    Input(s)
        - std_z (xarray.Dataset): standard deviations of residuals
        - min_x (xarray.Dataset): minimum values of unnnormalized states
        - std_x (xarray.Dataset): standard deviation of unnormalized states
        - mean_x (xarray.Dataset): mean of unnnormalized states
        - sampler (Sampler): sampler to use to generate residuals
    """

    def __init__(
        self,
        std_z: xarray.Dataset,
        min_x: xarray.Dataset,
        std_x: xarray.Dataset,
        mean_x: xarray.Dataset,
        sampler: Sampler,
    ):
        # Statistic attributes
        self.std_z = std_z
        self.min_x = min_x
        self.std_x = std_x
        self.mean_x = mean_x

        # Sampler attribute
        self.sampler = sampler

    def __call__(
        self,
        inputs: xarray.Dataset,
        target_template: xarray.Dataset,
        forcings: xarray.Dataset,
        observations: Optional[Array] = None,
        **kwargs,
    ) -> xarray.Dataset:
        """
        Draw a sample x^{k+1}_{(i)} from p(x^{k+1} | x^{k}) or p(x^{k+1} | x^{k}, y^{k+1})
        Input(s):
            - inputs (xarray.Dataset): unnormalized previous states x^{k} of the system with dimension (batch=1, time=2, lat=181, lon=360, levels=13)
            - target_template (xarray.Dataset): template of the target with dimension (batch=1, time=1, lat=181, lon=360, levels=13)
            - forcings (xarray.Dataset): unnormalized forcing terms used by the GenCast denoiser
            - observations (Optional[Array]): normalized observations from fictional or real weather stations with dimension (batch=1, num_stations * num_observed_variables)
        Returns
            - sample (xarray.Dataset): predicted residual (as xarray_jax) with dimension (batch=1, time=1, lat=181, lon=360, levels=13)
        """
        # 1) Clean the Sea Surface Temperature (SST) variable for inputs and forcings
        variable_to_clean = "sea_surface_temperature"
        if variable_to_clean in inputs.keys():
            clean_inputs = utils.clean_NaN(
                inputs, variable_to_clean, self.min_x[variable_to_clean]
            )
        else:
            clean_inputs = inputs
        if variable_to_clean in forcings.keys():
            clean_forcings = utils.clean_NaN(
                forcings, variable_to_clean, self.min_x[variable_to_clean]
            )
        else:
            clean_forcings = forcings

        # 2) Normalize inputs and forcings
        normalized_inputs = utils.normalize(clean_inputs, self.std_x, self.mean_x)
        normalized_forcings = utils.normalize(clean_forcings, self.std_x, self.mean_x)

        # 3) Use the sampler to predict a normalized residual
        hat_z = self.sampler(
            inputs=normalized_inputs,
            targets_template=target_template,
            forcings=normalized_forcings,
            observations=observations,
        )

        # 4) Unnormalize residual and add the previous unnormalized state of the system
        sample = utils.unnormalize_prediction_and_add_input(
            inputs=clean_inputs,
            norm_predictions=hat_z,
            std_z=self.std_z,
            std_x=self.std_x,
            mean_x=self.mean_x,
        )

        # 5) Reintroduce NaNs in the prediction
        sample = utils.reintroduce_nans(
            old_inputs=inputs, predictions=sample, variable=variable_to_clean
        )

        return sample

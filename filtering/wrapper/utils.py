# Libraries
import jax.numpy as jnp  # type: ignore
import logging
import numpy as np
import pandas as pd
import xarray

from jax import Array  # type: ignore
from typing import List, Optional, Union

from .graphcast import (
    model_utils,
    xarray_jax,
    xarray_tree,
)


def clean_NaN(
    dataset: xarray.Dataset, variable: str, fill_value: xarray.DataArray
) -> xarray.Dataset:
    """
    Replace NaN by an other value
    Input(s)
        - dataset (xarray.Dataset): dataset to clean
        - variable (str): variable to clean
        - fill_value (xarray.DataArray): value used as replacement of NaNs
    Returns
        - clean_dataset (xarray.Dataset): clean dataset without NaNs
    """
    data_array = dataset[variable]
    clean_dataset = dataset.assign({variable: data_array.fillna(fill_value)})
    return clean_dataset


def convert_xarray_to_jax(array_xarray: xarray.Dataset, jax_array: bool = True) -> Array:
    """
    Convert an xarray.Dataset to a jnp.ndarray
    Input(s)
        - array_xarray (xarray.Dataset): xarray to convert to jnp.ndarray with dimensions (batch=1, time=1, lat=181, lon=360, levels=13)
        - jax_array (bool): if true, xarray_jax are needed (otherwise an error is thrown)
    Returns
        - array_jnp (Array): jax array corresponding to the input xarray with dimensions (batch=1, lat=181, lon=360, num_channels)
    """
    array_jnp = model_utils.dataset_to_stacked(array_xarray)
    if jax_array:
        array_jnp = xarray_jax.jax_data(array_jnp)
    else:
        array_jnp = xarray_jax.unwrap_data(array_jnp)
    return array_jnp


def convert_jax_to_xarray(array_jnp: Array, template_dataset: xarray.Dataset) -> xarray.Dataset:
    """
    Convert an jnp.ndarray to a xarray.Dataset
    Input(s)
        - array_jnp (Array): jnp.ndarray to convert to xarray with dimensions (batch=1, lat=181, lon=360, num_channels)
        - template_dataset (xarray.Dataset): a template dataset used for the conversion
    Returns
        - array_xarray (xarray.Dataset): xarray corresponding to the input jnp.array with dimensions (batch=1, time=1, lat=181, lon=360, levels=13)
    """
    dims = ("batch", "lat", "lon", "channels")
    array_xarray = xarray_jax.DataArray(data=array_jnp, dims=dims)
    array_xarray = model_utils.stacked_to_dataset(array_xarray.variable, template_dataset)
    return array_xarray


def duplicate_xarray(array: xarray.Dataset, new_dim: str = "sample", n: int = 4):
    """
    Add a new dimension to an input xarray and duplicate it along this new dimension
    Input(s)
        - array (xarray.Dataset): input array to be duplicated
        - new_dim (str): name of the dimension
        - n (int): number of copies to do
    Returns
        - res (xarray.Dataset): output array with the new dimension
    """
    coord = pd.Index(np.arange(n), name=new_dim)
    res = xarray.concat([array] * n, dim=coord)
    return res


def draw_normalized_observations(
    x: xarray.Dataset,
    std_x: xarray.Dataset,
    mean_x: xarray.Dataset,
    sigma_y: Array,
    mask_satellite: Union[Array, None],
    mask_weather_stations: Union[Array, None],
    observed_variables_satellite: Union[List[str], None],
    observed_variables_weather_stations: Union[List[str], None],
) -> Array:
    """
    Draw normalized observations from p(hat{y}^{k} | x^{k})
    Input(s):
        - x (xarray.Dataset): state of the system with dimension (batch=1, time=1, lat=181, lon=360, levels=13)
        - std_x (xarray.Dataset): standard deviations of system states
        - mean_x (xarray.Dataset): means of system states
        - sigma_y (Array): diagonal covariance matrix of normalized observations with dimension (1, num_observed_variables)
        - mask_satellite (Union[Array, None]): boolean Array of dimension (181, 360) corresponding to satellite observations
        - mask_weather_stations (Union[Array, None]): boolean Array of dimension (181, 360) corresponding to ground observations
        - observed_variables_satellite (Union[List[str], None]): ordered list of variables observed by satellite
        - observed_variables_weather_stations (Union[List[str], None]): ordered list of variables observed by ground weather stations
    Returns
        observations (Array): a sample from p(hat{y}^{k} | x^{k})
    """
    # Normalize the input state
    hat_x = normalize(values=x, scales=std_x, locations=mean_x)

    # Extract observation from weather stations
    if (observed_variables_weather_stations is not None) and (mask_weather_stations is not None):
        obs_weather_stations = hat_x[observed_variables_weather_stations]
        obs_weather_stations = convert_xarray_to_jax(obs_weather_stations, jax_array=False)
        obs_weather_stations = obs_weather_stations[:, mask_weather_stations, :]
        obs_weather_stations = obs_weather_stations.reshape((
            obs_weather_stations.shape[0],
            -1,
        ))
    else:
        obs_weather_stations = jnp.array([[]])

    # Extract observation from satellite
    if (observed_variables_satellite is not None) and (mask_satellite is not None):
        obs_satellite = hat_x[observed_variables_satellite]
        obs_satellite = convert_xarray_to_jax(obs_satellite, jax_array=False)
        obs_satellite = obs_satellite[:, mask_satellite, :]
        obs_satellite = obs_satellite.reshape((
            obs_satellite.shape[0],
            -1,
        ))
    else:
        obs_satellite = jnp.array([[]])

    # Concatenate observations from ground stations and satellite
    observations = jnp.concatenate([obs_weather_stations, obs_satellite], axis=1)

    # Add noise to the observations
    noise = sigma_y * np.random.randn(*sigma_y.shape)
    observations += noise
    observations = jnp.array(observations)

    return observations


def normalize(
    values: xarray.Dataset, scales: xarray.Dataset, locations: Optional[xarray.Dataset]
) -> xarray.Dataset:
    """
    Normalize a dataset
    Input(s)
        - values (xarray.Dataset): dataset to normalize
        - scales (xarray.Dataset): std of variables
        - locations (xarray.Dataset): mean of variables
    """

    def normalize_array(array):
        if array.name is None:
            raise ValueError("Can't look up normalization constants because array has no name.")
        if locations is not None:
            if array.name in locations:
                array = array - locations[array.name].astype(array.dtype)
            else:
                logging.warning("No normalization location found for %s", array.name)
        if array.name in scales:
            array = array / scales[array.name].astype(array.dtype)
        else:
            logging.warning("No normalization scale found for %s", array.name)
        return array

    normalized_dataset = xarray_tree.map_structure(normalize_array, values)
    return normalized_dataset


def normalized_observation_covariance(
    std_x: xarray.Dataset,
    mask_satellite: Union[Array, None],
    mask_weather_stations: Union[Array, None],
    sigma_y_satellite: Union[Array, None],
    sigma_y_weather_stations: Union[Array, None],
    observed_variables_satellite: Union[List[str], None],
    observed_variables_weather_stations: Union[List[str], None],
) -> Array:
    """
    Get the covariance matrix of normalized observations with dimension (1, num_observed_variables)
    with:
        | num_observed_variables = num_observed_variables_ws + num_observed_variables_sat
        | num_observed_variables_ws = sum(mask_ws) * len(observed_variables_ws)
        | num_observed_variables_sat = sum(mask_sat) * len(observed_variables_sat) * 13
    Input(s)
        - std_x (xarray.Dataset): standard deviations of the system's state
        - mask_satellite (Union[Array, None]): boolean Array of dimension (181, 360) corresponding to satellite observations
        - mask_weather_stations (Union[Array, None]): boolean Array of dimension (181, 360) corresponding to ground observations
        - sigma_y_satellite (Union[Array, None]): covariance matrix of unnormalized satellite observations with dimension (len(observed_variables_sat), 13)
        - sigma_y_weather_stations (Union[Array, None]): covariance matrix of unnormalized ground observations with dimension (len(observed_variables_ws),)
        - observed_variables_satellite (Union[List[str], None]): ordered list of variables observed by satellite
        - observed_variables_weather_stations (Union[List[str], None]): oredered list of variables observed by ground weather stations
    Returns
        - sigma_y_hat: covariance matrix of normalized observations with dimension (1, num_observed_variables)
    """

    def normalized_weather_stations_covariance(
        std_x: xarray.Dataset,
        mask: Array,
        sigma_y: Array,
        observed_variables: List[str],
    ) -> Array:
        """
        Get the covariance matrix of normalized ground stations observations with dimension (1, num_observed_variables_ws)
        with:
            | num_observed_variables_ws = sum(mask) * len(observed_variables)
        Input(s):
            - std_x (xarray.Dataset): standard deviations of the system's state
            - mask (Array): boolean Array of dimension (181, 360) corresponding to ground observations
            - sigma_y (Array): covariance matrix of unnormalized ground stations observations with dimension (len(observed_variables),)
            - observed_variables (List[str]): ordered list of variables observed by ground weather stations
        Returns
            - sigma_y_hat: covariance matrix if normalized ground stations observations with dimension (1, num_observed_variables_ws)
        """
        # Check dimensions
        assert len(observed_variables) == sigma_y.shape[0]

        # Get std_{X|Y}
        std_xy = std_x[observed_variables]
        std_xy_array = jnp.concatenate([
            jnp.ravel(jnp.array(std_xy[v].values)) for v in sorted(std_xy.data_vars)
        ])

        # Get sigma_hat_y and the number of ground weather stations
        sigma_y_hat = sigma_y / (std_xy_array**2)
        num_weather_stations = int(sum(mask))

        # Duplicate sigma_y_hat
        sigma_y_hat = jnp.tile(sigma_y_hat, num_weather_stations)
        sigma_y_hat = sigma_y_hat[None, :]

        return sigma_y_hat

    def normalized_satellite_covariance(
        std_x: xarray.Dataset,
        mask: Array,
        sigma_y: Array,
        observed_variables: List[str],
    ) -> Array:
        """
        Get the covariance matrix of normalized satellite observations with dimension (1, num_observed_variables_sat)
        with:
            | num_observed_variables_sat = sum(mask) * len(observed_variables) * 13
        Input(s):
            - std_x (xarray.Dataset): standard deviations of the system's state
            - mask (Array): boolean Array of dimension (181, 360) corresponding to satellite observations
            - sigma_y (Array): covariance matrix of unnormalized satellite observations with dimension (len(observed_variables), 13)
            - observed_variables (List[str]): ordered list of variables observed by the satellite
        Returns
            - sigma_y_hat: covariance matrix if normalized satellite observations with dimension (1, num_observed_variables_satstd)
        """
        # Check dimensions
        assert len(observed_variables) == sigma_y.shape[0]

        # Get std_{X|Y}
        std_xy = std_x[observed_variables]
        std_xy_array = jnp.stack([
            jnp.ravel(jnp.array(std_xy[v].values)) for v in sorted(std_xy.data_vars)
        ])  # (n,13)

        # Get sigma_hat_y and the number of observed grid points
        sigma_y_hat = sigma_y / (std_xy_array**2)
        num_grid_points = int(sum(mask))

        # Reshape and duplicate sigma_y_hat
        sigma_y_hat = sigma_y_hat.reshape((-1,))
        sigma_y_hat = jnp.tile(sigma_y_hat, num_grid_points)
        sigma_y_hat = sigma_y_hat[None, :]

        return sigma_y_hat

    # Check that at least one type of observation is available
    satellite = (
        (mask_satellite is not None)
        and (sigma_y_satellite is not None)
        and (observed_variables_satellite is not None)
    )
    weather_stations = (
        (mask_weather_stations is not None)
        and (sigma_y_weather_stations is not None)
        and (observed_variables_weather_stations is not None)
    )
    assert satellite or weather_stations

    # Get the (diagonal) covariance matrix of normalized ground station observations
    if weather_stations:
        sigma_y_hat_ws = normalized_weather_stations_covariance(
            std_x=std_x,
            mask=mask_weather_stations,
            sigma_y=sigma_y_weather_stations,
            observed_variables=observed_variables_weather_stations,
        )
    else:
        sigma_y_hat_ws = None

    # Get the (diagonal) convariance matrix of normalized satellite observations
    if satellite:
        sigma_y_hat_sat = normalized_satellite_covariance(
            std_x=std_x,
            mask=mask_satellite,
            sigma_y=sigma_y_satellite,
            observed_variables=observed_variables_satellite,
        )
    else:
        sigma_y_hat_sat = None

    # Concatenate the two covariance matrix
    sigma_y_hat = jnp.concatenate([sigma_y_hat_ws, sigma_y_hat_sat], axis=1)

    return sigma_y_hat


def reintroduce_nans(
    old_inputs: xarray.Dataset,
    predictions: xarray.Dataset,
    variable: str,
) -> xarray.Dataset:
    """
    Reintroduce NaNs in the prediction
    Input(s)
        - old_inputs (xarray.Dataset): previous raw states of the system
        - prediction (xarray.Dataset): prediction obtained with the sampler
        - variable (str): variable to clean
    """
    if variable in predictions.keys():
        nan_mask = np.isnan(old_inputs[variable]).any(dim="time")
        with_nan_values = predictions[variable].where(~nan_mask, np.nan)
        predictions = predictions.assign({variable: with_nan_values})
    return predictions


def substract_input_and_normalize_target(
    inputs: xarray.Dataset,
    targets: xarray.Dataset,
    std_z: xarray.Dataset,
    std_x: xarray.Dataset,
    mean_x: xarray.Dataset,
) -> xarray.Dataset:
    """
    As the diffusion process produces a normalized residual, we generate a normalized residual using previous and next states
    Input(s)
        - inputs (xarray.Dataset): unnormalized states of the system at {k-2} and {k-1} used by the denoiser
        - target (xarray.Dataset): unnoramlized state of the system at time k
        - std_z (xarray.Dataset): std of residual
        - std_x (xarray.Dataset): std of variables
        - mean_x (xarray.Dataset): mean of variables
    """

    def _subtract_input_and_normalize_target(inputs, target, std_z, std_x, mean_x):
        if target.sizes.get("time") != 1:
            raise ValueError(
                "normalization.InputsAndResiduals only supports wrapping predictors that predict a single timestep."
            )
        if target.name in inputs:
            target_residual = target
            last_input = inputs[target.name].isel(time=-1)
            target_residual = target_residual - last_input
            return normalize(target_residual, std_z, None)
        else:
            return normalize(target, std_x, mean_x)

    return xarray_tree.map_structure(
        lambda t: _subtract_input_and_normalize_target(inputs, t, std_z, std_x, mean_x), targets
    )


def unnormalize(
    values: xarray.Dataset, scales: xarray.Dataset, locations: Optional[xarray.Dataset]
) -> xarray.Dataset:
    """
    Unnormalize a dataset
    Input(s)
        - values (xarray.Dataset): dataset to normalize
        - scales (xarray.Dataset): std of variables
        - locations (xarray.Dataset): mean of variables
    """

    def unnormalize_array(array):
        if array.name is None:
            raise ValueError("Can't look up normalization constants because array has no name.")
        if array.name in scales:
            array = array * scales[array.name].astype(array.dtype)
        else:
            logging.warning("No normalization scale found for %s", array.name)
        if locations is not None:
            if array.name in locations:
                array = array + locations[array.name].astype(array.dtype)
            else:
                logging.warning("No normalization location found for %s", array.name)
        return array

    return xarray_tree.map_structure(unnormalize_array, values)


def unnormalize_prediction_and_add_input(
    inputs: xarray.Dataset,
    norm_predictions: xarray.Dataset,
    std_z: xarray.Dataset,
    std_x: xarray.Dataset,
    mean_x: xarray.Dataset,
) -> xarray.Dataset:
    """
    As the diffusion process produces a normalized residual, we unnnormalized it and add it to the last input
    Input(s)
        - inputs (xarray.Dataset): unnormalized states of the system at {k-2} and {k-1} used by the conditional denoiser
        - norm_prediction (xarray.Dataset): output of a denoiser (conditional or not) corresponding to hat{z}_{k}
        - std_z (xarray.Dataset): std of residual
        - std_x (xarray.Dataset): std of variables
        - mean_x (xarray.Dataset): mean of variables
    Returns
        - output (xarray.Dataset): unnormalized prediction at next time k as an xarray.Dataset with dimension (batch_size, time = 1, lat, lon, levels)
    """

    def _unnormalize_prediction_and_add_input(inputs, norm_prediction, std_z, std_x, mean_x):
        if norm_prediction.sizes.get("time") != 1:
            raise ValueError(
                "normalization.InputsAndResiduals only supports predicting a single timestep."
            )
        if norm_prediction.name in inputs:
            prediction = unnormalize(norm_prediction, std_z, None)
            last_input = inputs[norm_prediction.name].isel(time=-1)
            prediction = prediction + last_input
            return prediction
        else:
            return unnormalize(norm_prediction, std_x, mean_x)

    output = xarray_tree.map_structure(
        lambda pred: _unnormalize_prediction_and_add_input(inputs, pred, std_z, std_x, mean_x),
        norm_predictions,
    )
    return output

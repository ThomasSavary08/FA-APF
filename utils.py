# Libraries
import jax.numpy as jnp  # type: ignore
import logging
import numpy as np
import pandas as pd
import xarray

from jax import Array  # type: ignore
from typing import List, Optional

from graphcast import model_utils, xarray_jax, xarray_tree


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
    sigma_y: Array,
    observed_variables: List[str],
) -> Array:
    """
    Get the covariance matrix of normalized observations
    Input(s)
        - std_x (xarray.Dataset): standard deviations of system states
        - sigma_y (Array): covariance matrix of unnormalized observations with dimension (len(observed_variables),)
        - observed_variables (List[str]): list of observed variables
    """
    std_xy = std_x[observed_variables]
    std_xy_array = jnp.concatenate([
        jnp.ravel(jnp.array(std_xy[v].values)) for v in sorted(std_xy.data_vars)
    ])
    sigma_hat_y = sigma_y / (std_xy_array**2)
    return sigma_hat_y


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

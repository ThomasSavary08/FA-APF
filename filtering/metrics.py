# Libraries
import dataclasses
import numpy as np
import os
import shutil
import tqdm
import xarray

from .wrapper.graphcast import checkpoint, data_utils, gencast


def ensemble_mean(filter_path: str, metric_path: str, num_steps: int, num_particles: int):
    """
    Compute the ensemble mean for a set of particles with equal weights (as this is the case for the FA-APF).
    Input(s)
        - filter_path (str): path to the result of the filter
        - metric_path (str): path to save the metrics
        - num_steps (int): number of assimilation steps performed by the filter
        - num_particles (int): number of particles used by the filter
    """
    # Check the number of steps
    num_folders = sum(
        os.path.isdir(os.path.join(filter_path, name)) and not name.startswith(".")
        for name in os.listdir(filter_path)
    )
    assert num_folders == (num_steps + 1)

    # Check the number of particles
    if filter_path[-1] == "/":
        step_path = filter_path + str("1/")
    else:
        step_path = filter_path + str("/1/")
    num_files = sum(
        os.path.isfile(os.path.join(step_path, name))
        and not name.startswith(".")
        and name != "y.npy"
        for name in os.listdir(step_path)
    )
    assert num_files == num_particles

    # Create the folders to save the ensemble means
    num_folders_metrics = sum(
        os.path.isdir(os.path.join(metric_path, name)) and not name.startswith(".")
        for name in os.listdir(metric_path)
    )
    if num_folders_metrics != num_steps:
        # Delete previous folders
        for name in os.listdir(metric_path):
            path = os.path.join(metric_path, name)
            if os.path.isdir(path) and not name.startswith("."):
                shutil.rmtree(path)

        # Create new ones
        for i in range(1, num_steps + 1):
            os.makedirs(os.path.join(metric_path, str(i)), exist_ok=True)

    # Compute the ensemble mean for each step
    for step in tqdm.tqdm(range(1, num_steps + 1)):
        # Load particles
        particles = []
        if filter_path[-1] == "/":
            step_path = filter_path + str(step) + "/"
        else:
            step_path = filter_path + "/" + str(step) + "/"
        for i in range(1, num_particles + 1):
            particle_path = step_path + str(i) + ".nc"
            with open(particle_path, "rb") as file:
                particle = xarray.load_dataset(file, decode_timedelta=True).compute()
            particle = particle.isel(time=[-1])
            particles.append(particle)

        # Compute ensemble mean
        particles = xarray.concat(particles, dim="batch")
        ensemble_mean = particles.mean(dim="batch", keepdims=True)

        # Save the result
        if metric_path[-1] == "/":
            file_path = metric_path + str(step) + str("/ensemble_mean.nc")
        else:
            file_path = metric_path + str("/") + str(step) + str("/ensemble_mean.nc")
        ensemble_mean.to_netcdf(file_path, format="NETCDF4", engine="netcdf4")


def skill(
    metric_path: str,
    gt_path: str,
    checkpoint_path: str,
    num_steps: int,
):
    """
    Compute the skill (ensemble mean RMSE)
    Input(s)
        - metric_path (str): path to the metrics
        - gt_path (str): path to the ground truth (an ERA5 trajectory)
        - checkpoint_path (str): path to the checkpoint
        - num_steps (int): number of assimilation steps performed by the filter
    """
    # Check the number of steps
    num_folders = sum(
        os.path.isdir(os.path.join(metric_path, name)) and not name.startswith(".")
        for name in os.listdir(metric_path)
    )
    assert num_folders == num_steps

    # Load the ground truth
    with open(gt_path, "rb") as file:
        data = xarray.load_dataset(file, decode_timedelta=True).compute()
    with open(checkpoint_path, "rb") as file:
        ckpt = checkpoint.load(file, gencast.CheckPoint)
    _, gt, _ = data_utils.extract_inputs_targets_forcings(
        data,
        target_lead_times=slice("12h", f"{(data.sizes['time'] - 2) * 12}h"),
        **dataclasses.asdict(ckpt.task_config),
    )

    # Compute the skill (RMSE of the ensemble mean) for each step
    for step in tqdm.tqdm(range(1, num_steps + 1)):
        # Load the ensemble mean and the correct ground truth step
        if metric_path[-1] == "/":
            ensemble_mean_path = metric_path + str(step) + str("/ensemble_mean.nc")
        else:
            ensemble_mean_path = metric_path + str("/") + str(step) + str("/ensemble_mean.nc")
        with open(ensemble_mean_path, "rb") as file:
            ensemble_mean = xarray.load_dataset(file, decode_timedelta=True).compute()
        gt_step = gt.isel(time=[int(step - 1)])

        # Compute the skill
        skill = (gt_step - ensemble_mean) ** 2
        skill = skill.mean(dim=["lat", "lon"])
        skill = skill.map(np.sqrt)

        # Save the result
        if metric_path[-1] == "/":
            file_path = metric_path + str(step) + str("/skill.nc")
        else:
            file_path = metric_path + str("/") + str(step) + str("/skill.nc")
        skill.to_netcdf(file_path, format="NETCDF4", engine="netcdf4")


def spread(
    filter_path: str,
    metric_path: str,
    num_steps: int,
    num_particles: int,
):
    """
    Compute the spread for a set of particles with equal weights (as this is the case for the FA-APF).
    Input(s)
        - filter_path (str): path to the result of the filter
        - metric_path (str): path to save the metrics
        - num_steps (int): number of assimilation steps performed by the filter
        - num_particles (int): number of particles used by the filter
    """
    # Check the number of steps
    num_folders_filter = sum(
        os.path.isdir(os.path.join(filter_path, name)) and not name.startswith(".")
        for name in os.listdir(filter_path)
    )
    num_folders_metrics = sum(
        os.path.isdir(os.path.join(metric_path, name)) and not name.startswith(".")
        for name in os.listdir(metric_path)
    )
    assert num_folders_filter == (num_steps + 1)
    assert num_folders_metrics == num_steps

    # Check the number of particles
    if filter_path[-1] == "/":
        step_path = filter_path + str("1/")
    else:
        step_path = filter_path + str("/1/")
    num_files = sum(
        os.path.isfile(os.path.join(step_path, name))
        and not name.startswith(".")
        and name != "y.npy"
        for name in os.listdir(step_path)
    )
    assert num_files == num_particles

    # Compute the spread for each step
    for step in tqdm.tqdm(range(1, num_steps + 1)):
        # Load the ensemble mean
        if metric_path[-1] == "/":
            ensemble_mean_path = metric_path + str(step) + str("/ensemble_mean.nc")
        else:
            ensemble_mean_path = metric_path + str("/") + str(step) + str("/ensemble_mean.nc")
        with open(ensemble_mean_path, "rb") as file:
            ensemble_mean = xarray.load_dataset(file, decode_timedelta=True).compute()

        # Load particles
        particles = []
        if filter_path[-1] == "/":
            step_path = filter_path + str(step) + "/"
        else:
            step_path = filter_path + str("/") + str(step) + "/"
        for i in range(1, num_particles + 1):
            particle_path = step_path + str(i) + ".nc"
            with open(particle_path, "rb") as file:
                particle = xarray.load_dataset(file, decode_timedelta=True).compute()
            particle = particle.isel(time=[-1])
            particle = (particle - ensemble_mean) ** 2
            particles.append(particle)

        # Compute the spread
        particles = xarray.concat(particles, dim="batch")
        spread = (1.0 / (num_particles - 1)) * particles.sum(dim="batch", keepdims=True)
        spread = spread.mean(dim=["lat", "lon"])
        spread = spread.map(np.sqrt)

        # Save the result
        if metric_path[-1] == "/":
            file_path = metric_path + str(step) + str("/spread.nc")
        else:
            file_path = metric_path + str("/") + str(step) + str("/spread.nc")
        spread.to_netcdf(file_path, format="NETCDF4", engine="netcdf4")


def compute_metrics(
    filter_path: str,
    gt_path: str,
    output_path: str,
    checkpoint_path: str,
    num_steps: int,
    num_particles: int,
):
    """
    Compute the metrics for (skill and spread) for a set of particles with equal weights
    Input(s)
        - filter_path (str): path to the result of the filter
        - gt_path (str): path to the ground truth (an ERA5 trajectory)
        - output_path (str): path to save the metrics
        - checkpoint_path (str): path to the checkpoint
        - num_steps (int): number of assimilation steps performed by the filter
        - num_particles (int): number of particles used by the filter
    """
    # Compute ensemble means
    print("Compute ensemble means...")
    ensemble_mean(
        filter_path=filter_path,
        metric_path=output_path,
        num_steps=num_steps,
        num_particles=num_particles,
    )
    print("")

    # Compute skills
    print("Compute skills...")
    skill(
        metric_path=output_path,
        gt_path=gt_path,
        checkpoint_path=checkpoint_path,
        num_steps=num_steps,
    )
    print("")

    # Compute spreads
    print("Compute spreads...")
    spread(
        filter_path=filter_path,
        metric_path=output_path,
        num_steps=num_steps,
        num_particles=num_particles,
    )

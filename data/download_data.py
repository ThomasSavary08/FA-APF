# Libraries
import os

from google.cloud import storage

# Authenticate with Google Cloud Storage
gcs_client = storage.Client.create_anonymous_client()
gcs_bucket = gcs_client.get_bucket("dm_graphcast")
dir_prefix = "gencast/"

###########################################
######### DOWNLOAD TRAJECTORIES ###########
###########################################

# Instanciate a folder for raw trajectories
local_dir = "./trajectories"
os.makedirs(local_dir, exist_ok=True)

# Define file names
dataset_file_name = "source-era5_date-2019-03-29_res-1.0_levels-13_steps-20.nc"
remote_filename = dir_prefix + f"dataset/{dataset_file_name}"
local_filename = os.path.join(local_dir, "2019_03_29_1.0_13_20.nc")

# Download the data
if not os.path.exists(local_filename):
    blob = gcs_bucket.blob(remote_filename)
    blob.download_to_filename(local_filename)

###########################################
#########    DOWNLOAD STATS     ###########
###########################################

# Instanciate a folder for statistics
local_dir = "./stats34"
os.makedirs(local_dir, exist_ok=True)

# Download std of z
dataset_filename = "stats/diffs_stddev_by_level.nc"
remote_filename = dir_prefix + dataset_filename
local_filename = os.path.join(local_dir, "std_z.nc")
if not os.path.exists(local_filename):
    blob = gcs_bucket.blob(remote_filename)
    blob.download_to_filename(local_filename)

# Download mean of x
dataset_filename = "stats/mean_by_level.nc"
remote_filename = dir_prefix + dataset_filename
local_filename = os.path.join(local_dir, "mean_x.nc")
if not os.path.exists(local_filename):
    blob = gcs_bucket.blob(remote_filename)
    blob.download_to_filename(local_filename)

# Download std of x
dataset_filename = "stats/stddev_by_level.nc"
remote_filename = dir_prefix + dataset_filename
local_filename = os.path.join(local_dir, "std_x.nc")
if not os.path.exists(local_filename):
    blob = gcs_bucket.blob(remote_filename)
    blob.download_to_filename(local_filename)

# Download min of x
dataset_filename = "stats/min_by_level.nc"
remote_filename = dir_prefix + dataset_filename
local_filename = os.path.join(local_dir, "min_x.nc")
if not os.path.exists(local_filename):
    blob = gcs_bucket.blob(remote_filename)
    blob.download_to_filename(local_filename)

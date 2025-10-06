# Libraries
from google.cloud import storage

from graphcast import checkpoint, gencast

# Authenticate with Google Cloud Storage
gcs_client = storage.Client.create_anonymous_client()
gcs_bucket = gcs_client.get_bucket("dm_graphcast")
dir_prefix = "gencast/"

# Download the checkpoint
model_name = "GenCast 1p0deg <2019.npz"
with gcs_bucket.blob(dir_prefix + f"params/{model_name}").open("rb") as file:
    ckpt = checkpoint.load(file, gencast.CheckPoint)

# Save the checkpoint on the disk
local_path = "./gencast_1deg.npz"
with open(local_path, "wb") as file:
    checkpoint.dump(file, ckpt)

# Try to load back the checkpoint
with open(local_path, "rb") as file:
    ckpt = checkpoint.load(file, gencast.CheckPoint)
print(ckpt)

# Libraries
import os
import warnings
import yaml

# Modify flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# Load the rollout function
from filtering import rollout

# Ignore warnings of GenCast (about sparsity)
warnings.filterwarnings("ignore")

# Load the configuration file
config_file_path = "./../config/rollout.yaml"
with open(config_file_path, "r") as f:
    config = yaml.safe_load(f)

# Do parallel rollout
rollout.generate_trajectories(
    num_samples=int(config["num_samples"]),
    output_path=str(config["output_path"]),
    data_path=str(config["data_path"]),
    checkpoint_path=str(config["checkpoint_path"]),
    sampler=str(config["sampler"]),
    sampler_config=dict(config["sampler_config"]),
    min_x_path=str(config["min_x_path"]),
    std_x_path=str(config["std_x_path"]),
    std_z_path=str(config["std_z_path"]),
    mean_x_path=str(config["mean_x_path"]),
)

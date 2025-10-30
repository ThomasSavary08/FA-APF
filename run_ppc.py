# Libraries
import os
import warnings
import yaml

# Modify flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# Load the PPC function
from filtering import posterior_predictive_check

# Ignore warnings of GenCast (about sparsity)
warnings.filterwarnings("ignore")

# Load the configuration file
config_file_path = './config/ppc.yaml'
with open(config_file_path, "r") as f:
    config = yaml.safe_load(f)

# Check the type of complex objects
assert isinstance(config["sampler_config"], dict)
assert isinstance(config["observed_variables_sat"], list)
assert isinstance(config["observed_variables_ws"], list)

# Do PPC
posterior_predictive_check.ppc(
    num_samples=int(config["num_samples"]),
    conditional_output_path=str(config["conditional_output_path"]),
    unconditional_output_path=str(config["unconditional_output_path"]),
    data_path=str(config["data_path"]),
    checkpoint_path=str(config["checkpoint_path"]),
    min_x_path=str(config["min_x_path"]),
    std_x_path=str(config["std_x_path"]),
    std_z_path=str(config["std_z_path"]),
    mean_x_path=str(config["mean_x_path"]),
    sampler=str(config["sampler"]),
    sampler_config=config["sampler_config"],
    mask_sat_path=str(config["mask_sat_path"]),
    mask_ws_path=str(config["mask_ws_path"]),
    observed_variables_sat=config["observed_variables_sat"],
    observed_variables_ws=config["observed_variables_ws"],
    sigma_y_sat_path=str(config["sigma_y_sat_path"]),
    sigma_y_ws_path=str(config["sigma_y_ws_path"]),
    solver=str(config["solver"]),
    max_iter=int(config["max_iter"]),
    tol=float(config["tol"]),
)

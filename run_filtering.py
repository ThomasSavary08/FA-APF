# Libraries
import os
import warnings
import yaml

# Modify flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# Load the PPC function
from filtering import fa_apf

# Ignore warnings of GenCast (about sparsity)
warnings.filterwarnings("ignore")

# Load the configuration file
config_file_path = "./config/filtering.yaml"
with open(config_file_path, "r") as f:
    config = yaml.safe_load(f)

# Do filtering wit the FA-APF
fa_apf.filtering(
    data_path=str(config['data_path']),
    output_path=str(config['output_path']),
    checkpoint_path=str(config['checkpoint_path']),
    N=int(config['num_samples']),
    N_thr_min=int(config['N_threshold_min']),
    N_thr_max=int(config['N_threshold_max']),
    alpha_init=float(config['alpha_init']),
    mask_sat_path=str(config['mask_sat_path']),
    mask_ws_path=str(config['mask_ws_path']),
    observed_variables_sat=list(config['observed_variables_sat']),
    observed_variables_ws=list(config['observed_variables_ws']),
    sigma_y_sat_path=str(config['sigma_y_sat_path']),
    sigma_y_ws_path=str(config['sigma_y_ws_path']),
    sampler=str(config['sampler']),
    sampler_config=dict(config['sampler_config']),
    std_z_path=str(config["std_z_path"]),
    min_x_path=str(config["min_x_path"]),
    std_x_path=str(config["std_x_path"]),
    mean_x_path=str(config["mean_x_path"]),
    max_iter_alpha=int(config['max_iter_alpha']),
    solver=str(config['solver']),
    max_iter_solver=int(config['max_iter']),
    tol_solver=float(config['tol']),
)

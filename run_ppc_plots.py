# Libraries
import yaml

from filtering import posterior_predictive_check

# Load the configuration file
config_file_path = "./config/ppc_plots.yaml"
with open(config_file_path, "r") as f:
    config = yaml.safe_load(f)

# Define the figsize
width = float(config["plot_size"]["width"])
height = float(config["plot_size"]["height"])
figsize = (width, height)

# Do PPC
posterior_predictive_check.plot_PPC(
    reference_path=str(config["data_path"]),
    checkpoint_path=str(config["checkpoint_path"]),
    conditional_path=str(config["conditional_path"]),
    unconditional_path=str(config["unconditional_path"]),
    output_path=str(config["output_path"]),
    mask_sat_path=str(config["mask_sat_path"]),
    mask_ws_path=str(config["mask_ws_path"]),
    variables=list(config["variables"]),
    stds=dict(config["stds"]),
    lat=int(config["lat"]),
    lon=int(config["lon"]),
    num_samples=int(config["num_samples"]),
    num_draws=int(config["num_draws"]),
    num_row=int(config["plot_size"]["num_row"]),
    num_col=int(config["plot_size"]["num_col"]),
    title=str(config["title"]),
    figsize=figsize,
    colors=list(config["colors"]),
    xlabels=list(config["xlabels"]),
)

# Libraries
import yaml

# Load the rollout function
from filtering import plots

# Load the configuration file
config_file_path = "./config/plots.yaml"
with open(config_file_path, "r") as f:
    config = yaml.safe_load(f)

# Ensure path are None or str
if (config["idealist_path"] is None) or (config["idealist_path"] == "None"):
    idealist_path = None
else:
    idealist_path = str(config["idealist_path"])
if (config["realistic_path"] is None) or (config["realistic_path"] == "None"):
    realistic_path = None
else:
    realistic_path = str(config["realistic_path"])

# Define the figsize for the different plots
width_first_plot = int(config["figsize_first_plot"]["width"])
height_first_plot = int(config["figsize_first_plot"]["height"])
figsize_first_plot = (width_first_plot, height_first_plot)

width_second_plot = int(config["figsize_second_plot"]["width"])
height_second_plot = int(config["figsize_second_plot"]["height"])
figsize_second_plot = (width_second_plot, height_second_plot)

# Do parallel rollout
plots.make_plots(
    num_steps=int(config["num_steps"]),
    unconditional_path=str(config["unconditional_path"]),
    idealist_path=idealist_path,
    realistic_path=realistic_path,
    gt_path=str(config["gt_path"]),
    checkpoint_path=str(config["checkpoint_path"]),
    output_path=str(config["output_path"]),
    variables_first_plot=list(config["variables_first_plot"]),
    num_row_first_plot=int(config["num_row_first_plot"]),
    num_col_first_plot=int(config["num_col_first_plot"]),
    title_first_plot=str(config["title_first_plot"]),
    figsize_first_plot=figsize_first_plot,
    ylabels_first_plot=list(config["ylabels_first_plot"]),
    variables_second_plot=list(config["variables_second_plot"]),
    levels_second_plot=list(config["levels_second_plot"]),
    title_second_plot=str(config["title_second_plot"]),
    title_third_plot=str(config["title_third_plot"]),
    figsize_second_plot=figsize_second_plot,
    ylabels_second_plot=list(config["ylabels_second_plot"]),
    filter_third_plot=str(config["filter_third_plot"]),
    times_steps_third_plot=list(config["time_steps_third_plot"]),
    times_steps_titles_third_plot=list(config["time_steps_titles_third_plot"]),
    variables_third_plot=list(config["variables_third_plot"]),
    levels_third_plot=list(config["levels_variable_third_plot"]),
)

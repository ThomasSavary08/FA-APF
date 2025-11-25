# Libraries
import yaml

from filtering import metrics

# Load the configuration file
config_file_path = "./../config/metrics.yaml"
with open(config_file_path, "r") as f:
    config = yaml.safe_load(f)

# Compute metrics
metrics.compute_metrics(
    filter_path=str(config["filter_path"]),
    gt_path=str(config["gt_path"]),
    output_path=str(config["output_path"]),
    checkpoint_path=str(config["checkpoint_path"]),
    num_steps=int(config["num_steps"]),
    num_particles=int(config["num_samples"]),
)

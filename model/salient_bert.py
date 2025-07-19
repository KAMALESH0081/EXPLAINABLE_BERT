import os
import yaml

# Path to your YAML config
config_path = os.path.join(os.path.dirname(__file__), "..", "config", "base_bert.yaml")

# Load YAML
with open(os.path.abspath(config_path)) as f:
    config = yaml.safe_load(f)

print(config)
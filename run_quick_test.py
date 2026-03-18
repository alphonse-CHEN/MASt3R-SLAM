"""Quick test: imports + config (no dataset)."""
import torch
import lietorch
import mast3r_slam
from mast3r_slam.config import load_config

print("torch:", torch.__version__)
print("lietorch: OK")
print("mast3r_slam: OK")
cfg = load_config("config/base.yaml")
print("config/base.yaml: OK")
print("All imports and config OK — ready for pipeline.")

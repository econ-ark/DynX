# Add the repository root to the Python path
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, repo_root)

from src.stagecraft.stage import Stage
from src.heptapod_b.core.api import initialize_model, initialize_stage
from src.heptapod_b.num.generate import compile_num
from src.heptapod_b.io.yaml_loader import load_config


if __name__ == "__main__":
    # Load the stage configuration
    config_file = os.path.join(current_dir, 'config', 'ConsInd_multi.yml')
    config = load_config(config_file)


    # Initialize the stage
    stage = Stage(config=config, master_config=None, init_rep=initialize_model, num_rep=compile_num)

    # Build the computational model
    stage.build_computational_model()

    print("Stage built")
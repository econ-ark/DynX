import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pprint
import yaml

# Add the modcraft root directory to the Python path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import the Stage class and the Heptapod modules
from dynx.stagecraft import Stage
from dynx.heptapodx.core.api import initialize_model as heptapod_model_init
from dynx.heptapodx.core.api import generate_numerical_model as heptapod_gen_num
from dynx.heptapodx.core.api import load_config

# Import the solver components from the local solvers package
try:
    from examples.economic_models.cons_indshock.solvers.whisperer import (
        operator_factory_in_situ,
        whisperer_external,
        plot_results,
        build_lifecycle_t1_stage as initialize_lifecycle_t1_stage
    )
except ImportError:
    print("Warning: Unable to import whisperer modules. Creating stub functions for testing.")
    
    # Create stub functions for testing if modules are not found
    def operator_factory_in_situ(stage):
        print("Stub: operator_factory_in_situ called")
        
    def whisperer_external(stage):
        print("Stub: whisperer_external called")
        
    def plot_results(stage, results_dcsn, results_arvl, image_dir, filename_suffix=""):
        print(f"Stub: plot_results called with suffix {filename_suffix}")
        
    def initialize_lifecycle_t1_stage(config_file, stage):
        print("Stub: initialize_lifecycle_t1_stage called")
        # Set minimal dummy values for testing
        stage.perches["cntn"].sol = {"vlu_cntn": np.zeros(10), "lambda_cntn": np.zeros(10)}
        stage.perches["arvl"].sol = {"vlu_arvl": np.zeros(10), "lambda_arvl": np.zeros(10)}
        stage.perches["dcsn"].sol = {"policy": np.zeros(10)}


def build_lifecycle_t1_stage(config_file):
    """Build a Stage for the penultimate period (T-1) of a lifecycle model.

    Creates and initializes a lifecycle model stage with proper terminal
    value functions and initial arrays for continuation perch.

    Parameters
    ----------
    config_file : str
        Path to the configuration YAML file

    Returns
    -------
    Stage
        The initialized lifecycle stage with proper perch data
    """
    # Create Stage, providing initialization and numerical representation modules
    config = load_config(config_file)

    t1_stage = Stage(
        name="LifecycleT1Stage", 
        init_rep=heptapod_model_init, 
        num_rep=heptapod_gen_num,
        config=config,
    )
    
    #t1_stage.load_config(config)

    # Build computational model using num_rep (heptapod_gen_num)
    t1_stage.build_computational_model()
    
    # Initialize the stage using the function from the 1DEGM model
    initialize_lifecycle_t1_stage(config_file, t1_stage)

    return t1_stage


if __name__ == "__main__":
    # Create directory for images if it doesn't exist
    image_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', 'images', 'consind'))
    if not os.path.exists(image_dir):
        print(f"Creating directory: {image_dir}")
        os.makedirs(image_dir)
    
    # ============================
    # Build In-situ Mode Stage
    # ============================
    print("\n=== Building In-situ Mode Stage ===")
    config_file = os.path.join(os.path.dirname(__file__), "ConsInd.yml")
    t1_stage = build_lifecycle_t1_stage(config_file)

    # Using operator_factory_in_situ from the 1DEGM model
    t1_stage.operator_factory = operator_factory_in_situ
    t1_stage.attach_operatorfactory_operators()

    t1_stage.solve_backward()
    results_dcsn_insitu = t1_stage.perches["dcsn"].sol
    results_arvl_insitu = t1_stage.perches["arvl"].sol

    print("In-situ mode: Solved backward")

    # Plot in-situ results using the plot_results function from the 1DEGM model
    plot_results(t1_stage, results_dcsn_insitu, results_arvl_insitu, image_dir, filename_suffix="_insitu")

    # ============================
    # Build External Mode Stage
    # ============================
    print("\n=== Building External Mode Stage ===")
    config_file = os.path.join(os.path.dirname(__file__), "ConsInd.yml")
    external_stage = build_lifecycle_t1_stage(config_file)
    # Set mode to external
    external_stage.model_mode = "external"
    
    # Call the external whisperer from the 1DEGM model
    whisperer_external(external_stage)
    
    results_dcsn_external = external_stage.perches["dcsn"].sol
    results_arvl_external = external_stage.perches["arvl"].sol
    
    print("External mode: Solved using external whisperer")
    
    # Plot external results
    plot_results(external_stage, results_dcsn_external, results_arvl_external, image_dir, filename_suffix="_external")
    
    # Compare results
    print("\n=== Comparing In-situ vs External Results ===")
    plt.figure(figsize=(12, 10))
    
    # Value function comparison
    plt.subplot(2, 1, 1)
    a_grid = t1_stage.dcsn_to_arvl.model.num.state_space.arvl.grids.a
    plt.plot(a_grid, results_arvl_insitu["vlu_arvl"], label="In-situ Value")
    plt.plot(a_grid, results_arvl_external["vlu_arvl"], linestyle='--', label="External Value")
    plt.title("Value Function Comparison")
    plt.xlabel("Assets (a)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    # Policy function comparison
    plt.subplot(2, 1, 2)
    m_grid = t1_stage.dcsn_to_arvl.model.num.state_space.dcsn.grids.m
    plt.plot(m_grid, results_dcsn_insitu["policy"], label="In-situ Policy")
    plt.plot(m_grid, results_dcsn_external["policy"], linestyle='--', label="External Policy")
    plt.title("Policy Function Comparison")
    plt.xlabel("Cash-on-Hand (m)")
    plt.ylabel("Consumption (c)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, "heptapod_lifecycle_comparison.png"))

    # ============================
    # Run time iteration using in-situ mode
    # ============================
    # Implement time iteration for 10 periods
    def run_time_iteration(t1_stage, num_periods=10):
        """Run time iteration for multiple periods.

        Creates a sequence of lifecycle stages, where each stage's continuation values
        are initialized using the previous stage's arrival values.

        Parameters
        ----------
        t1_stage : Stage
            The initial (T-1) stage that has already been solved
        num_periods : int, optional
            Number of periods to iterate, by default 10

        Returns
        -------
        list
            List of solved stages from T-1 to T-num_periods
        """
        # Store all stages in a list, starting with the T-1 stage
        all_stages = [t1_stage]

        # Iterate backward through time
        for t in range(2, num_periods + 1):
            # Create a new stage with the same configuration
            config_file = os.path.join(os.path.dirname(__file__), "ConsInd.yml")
            # Create a new stage providing both init and num reps upfront
            new_stage = Stage(
                name=f"LifecycleT{t}Stage",
                init_rep=heptapod_model_init,
                num_rep=heptapod_gen_num,
                config=config_file,
            )
            #new_stage.load_config(config_file)

            # Build computational model using num_rep
            new_stage.build_computational_model()

            new_stage.operator_factory = operator_factory_in_situ
            new_stage.attach_operatorfactory_operators()

            # Attach the previous arrival data directly to the new stage's continuation perch
            new_stage.perches["cntn"].sol = {
                "vlu_cntn": all_stages[-1].perches["arvl"].sol["vlu_arvl"],
                "lambda_cntn": all_stages[-1].perches["arvl"].sol["lambda_arvl"],
            }

            new_stage.solve_backward()

            # Add the solved stage to our list
            all_stages.append(new_stage)

        return all_stages

    # Plot value functions and policies across multiple periods
    def plot_multiperiod_results(all_stages, image_dir):
        """Plot value functions and policies across multiple periods.

        Parameters
        ----------
        all_stages : list
            List of solved stages
        image_dir : str
            Directory where to save the output images
        """
        plt.figure(figsize=(12, 10))

        # 1. Value functions at arrival perch
        plt.subplot(2, 1, 1)
        for i, stage in enumerate(all_stages):
            period = i + 1
            a_grid = stage.dcsn_to_arvl.model.num.state_space.arvl.grids.a
            vlu_arvl = stage.perches["arvl"].sol["vlu_arvl"]
            plt.plot(a_grid, vlu_arvl, label=f"T-{period}")

        plt.title("Value Functions at Arrival Perch")
        plt.xlabel("Assets (a)")
        plt.ylabel("Value")
        plt.legend(loc="upper left", ncol=2)
        plt.grid(True)

        # 2. Consumption policies
        plt.subplot(2, 1, 2)
        for i, stage in enumerate(all_stages):
            period = i + 1
            m_grid = stage.dcsn_to_arvl.model.num.state_space.dcsn.grids.m  # Get grid from mover
            policy = stage.perches["dcsn"].sol["policy"]
            plt.plot(m_grid, policy, label=f"T-{period}")

        plt.title("Consumption Policies")
        plt.xlabel("Cash-on-Hand (m)")
        plt.ylabel("Consumption (c)")
        plt.legend(loc="upper left", ncol=2)
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, "heptapod_lifecycle_multiperiod_results.png"))

    # Run the time iteration and store all stages
    all_stages = run_time_iteration(t1_stage, num_periods=10)

    # Print summary of solved stages
    print("\nTime iteration complete. Solved stages:")
    for i, stage in enumerate(all_stages):
        period = i + 1
        print(f"Stage T-{period}: Solved.")  # Simplified message

    # Plot results across periods
    plot_multiperiod_results(all_stages, image_dir)
    print(f"\nImages saved in: {image_dir}")

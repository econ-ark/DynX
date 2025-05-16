"""
StageCraft Example: State Flag Workflow
---------------------------------------

This example demonstrates various workflow patterns for a Stage with
backward solution and forward simulation using state flags.

These flag attributes enable more sophisticated control of
the execution pipeline and provide more informative debugging.
"""

import os
import sys
import numpy as np

# Assuming plugins are adjacent or in path
# Need to adjust path if plugins are located elsewhere relative to examples
try:
    # This assumes a specific structure where plugins might be siblings
    # Adjust if your project structure is different
    plugin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'plugins', 'heptapod', 'src'))
    if os.path.exists(plugin_path):
         sys.path.insert(0, plugin_path)
    # Attempt to import the necessary modules from heptapod
    from heptapod import model_init, gen_num
except ImportError as e:
    print(f"Error importing Heptapod plugins: {e}")
    print(f"Looked in: {plugin_path}")
    print("Please ensure the Heptapod plugins (model_init, gen_num) are accessible in the Python path.")
    sys.exit(1)

# Try different import approaches to make the script runnable from various locations
try:
    # When running from the project root or if package is installed
    from stagecraft import Stage
except ImportError:
    try:
        # When running from examples directory
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from src.stagecraft import Stage
    except ImportError:
        raise ImportError(
            "Unable to import stagecraft. Make sure you're either:\n"
            "1. Running from the project root directory\n"
            "2. Have installed the package with 'pip install -e .'\n"
            "3. Have added the project root to your PYTHONPATH"
        )

# --- Configuration ---\n# Assumes the config file is in the same directory as this script
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "example_config.yml")

# --- Helper Function ---
def print_status(stage_instance):
    """Prints the current status flags of the stage."""
    print("-" * 30)
    print(f"Stage: {stage_instance.name}")
    print("Current Status Flags:")
    if hasattr(stage_instance, 'status_flags') and isinstance(stage_instance.status_flags, dict):
        # Sort flags for consistent output
        sorted_flags = sorted(stage_instance.status_flags.items())
        for flag, value in sorted_flags:
            print(f"  - {flag}: {value}")
    else:
        print("  Status flags not available or not a dictionary.")
    print("-" * 30)

# --- Main Demo ---
def main():
    print("Starting Status Flags Demo...")

    # 1. Create Stage instance
    print("\n[1] Creating Stage instance...")
    # Provide the Heptapod plugins for initialization and numerical model generation
    stage = Stage(name="StatusDemoStage", init_rep=model_init, num_rep=gen_num)
    print_status(stage)
    # Expected: All flags False initially.

    # 2. Load Configuration
    print("\n[2] Loading configuration...")
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Config file not found at {CONFIG_FILE}")
        return
    try:
        stage.load_config(CONFIG_FILE)
        print("Configuration loaded successfully.")
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    print_status(stage)
    # Expected: initialized=True, all_perches_initialized=True, all_movers_initialized=True (based on config)
    # Other flags should remain False.

    # 3. Build Computational Model
    print("\n[3] Building computational model...")
    try:
        success = stage.build_computational_model()
        if success:
            print("Computational model built successfully.")
        else:
            print("Failed to build computational model.")
            # Attempting to proceed might fail later, but let's see status flags
            # return # Option to stop if build fails
    except Exception as e:
        print(f"Error building computational model: {e}")
        return
    print_status(stage)
    # Expected: compiled=True, all_perches_compiled=True, all_movers_compiled=True.
    # Solvable, solved, simulated, portable should still be False.

    # 4. Set Initial Data for Solvability
    # According to Stage._check_solvability, we need cntn.up and arvl.down
    # to be non-None for the 'solvable' flag to become True.
    print("\n[4] Setting dummy initial data for solvability...")
    try:
        # Use a simple numpy array. The exact shape/value might depend on the model,
        # but for demonstrating the flag, non-None is the key.
        dummy_data_up = np.array([1.0])
        dummy_data_down = np.array([0.0])

        if stage.cntn:
            stage.cntn.up = dummy_data_up
            print(f"Set dummy data for cntn.up: {type(stage.cntn.up)}")
        else:
            print("Warning: cntn perch not found.")

        if stage.arvl:
            stage.arvl.down = dummy_data_down
            print(f"Set dummy data for arvl.down: {type(stage.arvl.down)}")
        else:
            print("Warning: arvl perch not found.")

    except Exception as e:
        print(f"Error setting initial data: {e}")
        # Proceeding might fail, but allows observing flags

    # 5. Check Solvability
    print("\n[5] Checking solvability...")
    try:
        # Call the internal check method to update the flag
        stage._check_solvability()
        print(f"Solvability Check Completed. Flag 'solvable': {stage.status_flags['solvable']}")
    except Exception as e:
        print(f"Error checking solvability: {e}")
        # Continue to see if solve catches it later
    print_status(stage)
    # Expected: solvable=True (if dummy data was set correctly on cntn and arvl).

    # 6. Solve Backward
    print("\n[6] Solving backward...")
    if not stage.status_flags['solvable']:
        print("Stage not solvable, skipping backward solve.")
    else:
        try:
            # Note: Stage.solve() calls solve_backward() and solve_forward()
            # Calling individually to observe flag changes more granularly.
            stage.solve_backward()
            print("Backward solve completed.")
        except RuntimeError as e:
             print(f"RuntimeError during backward solve: {e}")
        except Exception as e:
            print(f"Unexpected error during backward solve: {e}")
    print_status(stage)
    # Expected: solved=True (if backward pass ran and arvl.up is now non-None).

    # 7. Solve Forward
    print("\n[7] Solving forward...")
    if not stage.status_flags['solvable']:
        print("Stage not solvable, skipping forward solve.")
    # Forward solve usually depends on backward solve being done (having 'up' data)
    # Although the parent CircuitBoard.solve_forward checks for initial 'down' data.
    elif not stage.status_flags['solved']:
         print("Backward solve did not complete successfully ('solved' is False), skipping forward solve.")
    else:
        try:
            stage.solve_forward()
            print("Forward solve completed.")
        except RuntimeError as e:
             print(f"RuntimeError during forward solve: {e}")
        except Exception as e:
            print(f"Unexpected error during forward solve: {e}")
    print_status(stage)
    # Expected: simulated=True (if forward pass ran and cntn.down is now non-None).

    # 8. Final Status Check
    print("\n[8] Final Status Check:")
    print_status(stage)
    print(f"\nNote: 'portable' flag remains {stage.status_flags['portable']}.")
    print("It typically becomes True only after operators (computational components) ")
    print("are attached to all movers, e.g., via stage.attach_whisperer_operators(), ")
    print("which was not performed in this demo.")

    print("\nStatus Flags Demo finished.")

if __name__ == "__main__":
    main()
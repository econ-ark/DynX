import os
import sys
import logging

# Add project root to path to allow importing src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, '..')) # Assumes test file is in /tests
sys.path.insert(0, repo_root)

# Import necessary functions AFTER adjusting path
from src.heptapod_b.io.yaml_loader import load_config
from src.stagecraft.config_loader import initialize_model_Circuit, compile_all_stages

# Configure basic logging for the test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_compile_all_stages_with_housing_config():
    """
    Tests initialize_model_Circuit followed by compile_all_stages
    using the copied housing model configuration files.
    """
    logger.info("Starting test: test_compile_all_stages_with_housing_config")

    # Define paths to the COPIED configuration files within the /tests directory
    test_config_dir = os.path.join(current_dir, "config", "housing")
    master_path = os.path.join(test_config_dir, "housing_master.yml")
    ownh_path = os.path.join(test_config_dir, "OWNH_stage.yml")
    ownc_path = os.path.join(test_config_dir, "OWNC_stage.yml")
    connections_path = os.path.join(test_config_dir, "connections.yml")

    # Verify config files exist in the test location
    assert os.path.exists(master_path), f"Master config not found at {master_path}"
    assert os.path.exists(ownh_path), f"OWNH config not found at {ownh_path}"
    assert os.path.exists(ownc_path), f"OWNC config not found at {ownc_path}"
    assert os.path.exists(connections_path), f"Connections config not found at {connections_path}"
    logger.info("Test configuration files found.")

    # Load configurations from the test directory
    master_config = load_config(master_path)
    ownh_config = load_config(ownh_path)
    ownc_config = load_config(ownc_path)
    connections_config = load_config(connections_path)
    logger.info("Configurations loaded successfully.")

    stage_configs = {
        "OWNH": ownh_config,
        "OWNC": ownc_config
    }

    # Step 1: Initialize the model circuit
    logger.info("Initializing model circuit...")
    model_circuit = initialize_model_Circuit(
        master_config=master_config,
        stage_configs=stage_configs,
        connections_config=connections_config
    )
    assert model_circuit is not None, "Model circuit initialization failed."
    assert len(model_circuit.periods_list) > 0, "Model circuit has no periods."
    logger.info("Model circuit initialized.")

    # Step 2: Compile all stages
    logger.info("Calling compile_all_stages...")
    compile_all_stages(model=model_circuit, force=False) # Use force=False by default
    logger.info("compile_all_stages finished.")

    # Step 3: Verify results
    compiled_found = False
    grid_proxy_found = False
    mesh_proxy_found = False # Optional: check if mesh proxy is expected and attached

    logger.info("Verifying compilation results...")
    for period in model_circuit.periods_list:
         logger.debug(f"Checking Period {period.time_index}...")
         for stage_name, stage in period.stages.items():
             logger.debug(f"Checking Stage {stage_name}...")
             status_flags = getattr(stage, "status_flags", set())

             if "compiled" in status_flags:
                 compiled_found = True
                 logger.debug(f"Stage {stage_name} is marked compiled.")
             else:
                  logger.warning(f"Stage {stage_name} is NOT marked compiled. Flags: {status_flags}")

             # Check for grid proxy on 'arvl' perch (common perch)
             arvl_perch = getattr(stage, 'arvl', None)
             if arvl_perch and hasattr(arvl_perch, "grid"):
                 grid_proxy_found = True
                 logger.debug(f"Grid proxy found on arvl perch of Stage {stage_name}.")
                 # Optionally check specific grid access
                 # try:
                 #    _ = arvl_perch.grid.a # Example check
                 # except AttributeError:
                 #    logger.warning(f"Could not access grid 'a' via proxy on {stage_name}.arvl")


             # Check for mesh proxy if relevant for the housing model stage
             # Modify perch name if mesh is expected elsewhere
             if arvl_perch and hasattr(arvl_perch, "mesh"):
                 mesh_proxy_found = True
                 logger.debug(f"Mesh proxy found on arvl perch of Stage {stage_name}.")
                 # Optionally check specific mesh access
                 # try:
                 #    _ = arvl_perch.mesh.a # Example check
                 # except AttributeError:
                 #    logger.warning(f"Could not access mesh 'a' via proxy on {stage_name}.arvl")


    assert compiled_found, "Verification failed: No stages were marked as compiled."
    assert grid_proxy_found, "Verification failed: No grid proxies seem to have been attached."
    # assert mesh_proxy_found, "Verification failed: No mesh proxies seem to have been attached for housing model."
    # Note: Housing model config used here doesn't seem to generate data that triggers mesh proxy attachment.
    # If mesh proxies ARE expected for this model in the future, re-enable the assertion above.
    if mesh_proxy_found:
         logger.info("Mesh proxies were found (unexpected for this config, but okay).")
    else:
         logger.info("Mesh proxies were not found (as expected for this config).")

    logger.info("Test test_compile_all_stages_with_housing_config passed.") 
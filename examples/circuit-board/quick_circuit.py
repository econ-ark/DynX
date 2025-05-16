"""
StageCraft 1.2.0 Example: Quick Circuit Creation
============================================================
This example demonstrates how to create and solve a simple circuit using
StageCraft 1.2.0 terminology and features.



Todo
----
Needs updating with latest version. 
"""

import numpy as np
import sys
import os

# Try different import approaches to make the script runnable from various locations
try:
    # When running from the project root or if package is installed
    from src.stagecraft import CircuitBoard, Perch
except ImportError:
    try:
        # When running from examples directory with src structure
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from src.stagecraft import CircuitBoard, Perch
    except ImportError:
        raise ImportError(
            "Unable to import stagecraft. Make sure you're either:\n"
            "1. Running from the project root directory\n"
            "2. Have installed the package with 'pip install -e .'\n"
            "3. Have added the project root to your PYTHONPATH"
        )


def main():
    print("StageCraft 1.2.0 Example: Quick Circuit Creation")
    print("============================================================")
    
    # Define a simple computational method for backward solving
    def square(data):
        """Square function - used for backward solving."""
        comp = data.get("comp")
        if comp is not None:
            return {"comp": comp**2}
        return {}
    
    # Define a simple computational method for forward simulation
    def add_values(data):
        """Add sim and comp values - used for forward simulation."""
        sim = data.get("sim")
        comp = data.get("comp")
        if sim is not None and comp is not None:
            return {"sim": sim + comp}
        return {}
    
    print("\nCreating a simple 2-perch circuit manually:")
    
    # Create the circuit
    circuit = CircuitBoard(name="SimpleCircuit")
    
    # Add perches
    circuit.add_perch(Perch("input", {"comp": None, "sim": None}))
    circuit.add_perch(Perch("output", {"comp": None, "sim": None}))
    
    # Add movers
    # Backward mover (for comp calculation)
    circuit.add_mover(
        source_name="output", 
        target_name="input",
        source_key="comp", 
        target_key="comp",
        edge_type="backward"
    )
    
    # Forward mover (for sim propagation)
    circuit.add_mover(
        source_name="input", 
        target_name="output",
        source_keys=["comp", "sim"], 
        target_key="sim",
        edge_type="forward"
    )
    
    # Print lifecycle flags
    print("\nLIFECYCLE FLAGS AFTER MOVERS CREATION:")
    print(f"has_empty_perches: {circuit.has_empty_perches}")
    print(f"has_model: {circuit.has_model}")
    print(f"movers_backward_exist: {circuit.movers_backward_exist}")
    print(f"is_portable: {circuit.is_portable}")
    print(f"is_solvable: {circuit.is_solvable}")
    
    # Create maps
    square_map = {
        "operation": "square",
        "parameters": {}
    }
    
    add_map = {
        "operation": "add",
        "parameters": {}
    }
    
    # Set maps
    circuit.set_mover_map("output", "input", "backward", square_map)
    circuit.set_mover_map("input", "output", "forward", add_map)
    
    # Define comp factory
    def comp_factory(data):
        """Convert maps to computational methods."""
        map_data = data.get("map", {})
        operation = map_data.get("operation")
        
        if operation == "square":
            return square
        elif operation == "add":
            return add_values
        
        # Default fallback
        return lambda data: {}
    
    # Create comps from maps
    circuit.create_comps_from_maps(comp_factory)
    
    # Finalize the model
    circuit.finalize_model()
    
    print("\nLIFECYCLE FLAGS AFTER MODEL FINALIZATION:")
    print(f"has_empty_perches: {circuit.has_empty_perches}")
    print(f"has_model: {circuit.has_model}")
    print(f"movers_backward_exist: {circuit.movers_backward_exist}")
    print(f"is_portable: {circuit.is_portable}")
    print(f"is_solvable: {circuit.is_solvable}")
    
    # Initialize perch values
    circuit.set_perch_data("input", {"comp": 3.0, "sim": 2.0})
    
    print("\nPERCH VALUES AFTER INITIALIZATION:")
    print(f"input.comp = {circuit.get_perch_data('input', 'comp')}")
    print(f"input.sim = {circuit.get_perch_data('input', 'sim')}")
    print(f"output.comp = {circuit.get_perch_data('output', 'comp')}")
    print(f"output.sim = {circuit.get_perch_data('output', 'sim')}")
    
    print("\nLIFECYCLE FLAGS AFTER INITIALIZATION:")
    print(f"has_empty_perches: {circuit.has_empty_perches}")
    print(f"has_model: {circuit.has_model}")
    print(f"movers_backward_exist: {circuit.movers_backward_exist}")
    print(f"is_portable: {circuit.is_portable}")
    print(f"is_solvable: {circuit.is_solvable}")
    
    # Solve the circuit
    print("\nSolving the circuit...")
    
    # For backward solving, set output.comp value
    circuit.set_perch_data("output", {"comp": 9.0})
    
    # Solve the circuit with automatic solving
    circuit.solve()
    
    # Print the results
    input_comp = circuit.get_perch_data("input", "comp")
    output_comp = circuit.get_perch_data("output", "comp")
    
    input_sim = circuit.get_perch_data("input", "sim")
    output_sim = circuit.get_perch_data("output", "sim")
    
    print("\nCircuit solved successfully!")
    print("\nComp (backward) results:")
    print(f"input.comp = {input_comp}")   # Initial value: 3.0
    print(f"output.comp = {output_comp}") # Should be 3.0^2 = 9.0
    
    print("\nSim (forward) results:")
    print(f"input.sim = {input_sim}")     # Initial value: 2.0
    print(f"output.sim = {output_sim}")   # Should be 2.0 + 9.0 = 11.0
    
    # Print the lifecycle flags
    print("\nCircuit board status after solution:")
    print(f"has_model: {circuit.has_model}")
    print(f"movers_backward_exist: {circuit.movers_backward_exist}")
    print(f"is_portable: {circuit.is_portable}")
    print(f"is_solvable: {circuit.is_solvable}")
    print(f"is_solved: {circuit.is_solved}")
    print(f"is_simulated: {circuit.is_simulated}")
    
    print("\nKey StageCraft 1.2.0 Features:")
    print("- Perch (formerly Node): Stores data like comp and sim values")
    print("- CircuitBoard (formerly Graph): Organizes perches and movers")
    print("- Mover (formerly Edge): Contains operations between perches")
    print("- comp (formerly function): Represents policy/decision functions")
    print("- sim (formerly distribution): Represents state distributions")
    print("- Lifecycle flags: Track the state of the circuit (model, portable, solved, etc.)")


if __name__ == "__main__":
    main() 
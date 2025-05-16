"""
StageCraft Example: Quick Workflow with High-Level API
------------------------------------------------------
This example demonstrates the use of StageCraft's high-level API
to create and solve a circuit following the workflow:

1. Circuit Creation: Create the circuit board structure with perches and movers
2. Model Finalization: Finalize the model once all components are added
3. Portability Check: Make the circuit portable for serialization
4. Initialization: Provide initial values
5. Solution: Execute operations

The example shows how to use the `create_and_solve_circuit` function, 
which handles all workflow steps in one convenient call.

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
    from src.stagecraft import create_and_solve_circuit
except ImportError:
    try:
        # When running from examples directory
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from src.stagecraft import create_and_solve_circuit
    except ImportError:
        raise ImportError(
            "Unable to import stagecraft. Make sure you're either:\n"
            "1. Running from the project root directory\n"
            "2. Have installed the package with 'pip install -e .'\n"
            "3. Have added the project root to PYTHONPATH"
        )

def main():
    print("StageCraft 1.2.0 Quick Workflow Example")
    print("----------------------------------")
    
    # Define operations (simple functions)
    def square(data):
        """Square the up value (backward operation)"""
        # Handle both dictionary inputs and direct scalar inputs
        if isinstance(data, dict):
            up_value = data.get("up")
        else:
            # If data is directly a scalar value
            up_value = data
            
        if up_value is not None:
            return {"up": up_value**2}
        return {}
    
    def add_ten(data):
        """Add 10 to the down value (forward operation)"""
        # Handle both dictionary inputs and direct scalar inputs
        if isinstance(data, dict):
            down_value = data.get("down")
        else:
            # If data is directly a scalar value
            down_value = data
            
        if down_value is not None:
            return {"down": down_value + 10}
        return {}
    
    # Create and solve a circuit in one high-level call
    # This handles all workflow steps internally
    circuit = create_and_solve_circuit(
        name="QuickCircuit",
        
        # Circuit Creation - Define perches (formerly nodes)
        nodes=[
            {"id": "A", "data_types": {"up": None, "down": None}},
            {"id": "B", "data_types": {"up": None, "down": None}},
            {"id": "C", "data_types": {"up": None, "down": None}}
        ],
        
        # Configuration - Define movers (formerly edges) with operations
        edges=[
            # Backward mover: B → A
            {"source": "B", "target": "A", 
             "operation": square, 
             "source_key": "up", "target_key": "up",
             "edge_type": "backward"},
            
            # Forward mover: A → B
            {"source": "A", "target": "B", 
             "operation": add_ten,
             "source_key": "down", "target_key": "down", 
             "edge_type": "forward"},
            
            # Forward mover: B → C
            {"source": "B", "target": "C", 
             "operation": add_ten,
             "source_key": "down", "target_key": "down",
             "edge_type": "forward"}
        ],
        
        # Initialization - Set initial values
        initial_values={
            "A": {"up": 0, "down": 5},  # Initial terminal value for backward solving and initial down
            "B": {"up": 5, "down": None},  # Initial up value at perch B
            "C": {"up": None, "down": None}  # Initialize perch C with None
        }
    )
    
    # Verify circuit lifecycle flags
    print(f"\nCircuit board lifecycle status:")
    print(f"- has_model: {circuit.has_model}")
    print(f"- is_portable: {circuit.is_portable}")
    print(f"- is_solvable: {circuit.is_solvable}")
    print(f"- is_solved: {circuit.is_solved}")
    print(f"- is_simulated: {circuit.is_simulated}")
    
    # Access results (using get_perch_data instead of get_node_data)
    a_up = circuit.get_perch_data("A", "up")
    a_down = circuit.get_perch_data("A", "down")
    b_up = circuit.get_perch_data("B", "up")
    b_down = circuit.get_perch_data("B", "down")
    c_up = circuit.get_perch_data("C", "up")
    c_down = circuit.get_perch_data("C", "down")
    
    print("\nRESULTS:")
    print(f"Perch A - up: {a_up}, down: {a_down}")   # up: 5² = 25, down: None initially
    print(f"Perch B - up: {b_up}, down: {b_down}")   # up: 5, down: A's down + 10 = None + 10
    print(f"Perch C - up: {c_up}, down: {c_down}")   # up: None, down: B's down + 10
    
    print("\nHow it works internally:")
    print("1. Circuit Creation: Creates perches A, B, C and connects them with movers")
    print("2. Model Finalization: Finalizes the circuit board model")
    print("3. Portability: Makes the circuit portable for serialization")
    print("4. Initialization: Sets initial value for perch B (up=5)")
    print("5. Solution: Executes operations in order:")
    print("   - Backward: B (up=5) → A, applying square: 5² = 25")
    print("   - Forward: A (down=None) → B, applying add_ten (if down were not None)")
    print("   - Forward: B (down=None) → C, applying add_ten (if down were not None)")

if __name__ == "__main__":
    main() 
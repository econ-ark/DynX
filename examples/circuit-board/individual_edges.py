"""
StageCraft Example: Individual Mover Execution
----------------------------------------------
This example demonstrates how to execute individual movers in a circuit
and work directly with the comp and sim attributes of perches.

Todo
------
Needs updating with latest version. 
"""

import numpy as np
import sys
import os

# Try different import approaches to make the script runnable from various locations
try:
    # When running from the project root or if package is installed
    from stagecraft import CircuitBoard
    from stagecraft import Perch
except ImportError:
    try:
        # When running from examples directory with src structure
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from src.stagecraft import CircuitBoard
        from src.stagecraft import Perch
    except ImportError:
        raise ImportError(
            "Unable to import stagecraft. Make sure you're either:\n"
            "1. Running from the project root directory\n"
            "2. Have installed the package with 'pip install -e .'\n"
            "3. Have added the project root to your PYTHONPATH"
        )


def matrix_transform(matrix, vector):
    """Transform a matrix using a vector outer product"""
    v_column = vector.reshape(-1, 1)
    outer_product = v_column @ v_column.T
    return matrix + 0.1 * (matrix @ outer_product)


def main():
    print("StageCraft 1.2.0 Individual Mover Execution Example")
    print("---------------------------------------------------")
    
    #--------------------------------------------------
    # 1. CIRCUIT CREATION
    #--------------------------------------------------
    print("\n1. CIRCUIT CREATION")
    
    # Create circuit
    circuit = CircuitBoard(name="IndividualMovers")
    print(f"Circuit board created: {circuit.name}")
    
    # Add perches
    circuit.add_perch(Perch("perch_0", {"comp": None, "sim": None, "vector": None, "matrix": None}))
    circuit.add_perch(Perch("perch_1", {"comp": None, "sim": None, "vector": None, "matrix": None}))
    circuit.add_perch(Perch("perch_2", {"comp": None, "sim": None, "vector": None, "matrix": None}))
    print(f"Added 3 perches to the circuit board")
    
    # Add movers (without maps initially)
    circuit.add_mover(
        source_name="perch_1", 
        target_name="perch_0",
        source_key="vector", 
        target_key="vector",
        edge_type="backward"
    )
    
    circuit.add_mover(
        source_name="perch_0", 
        target_name="perch_1",
        source_keys=["matrix", "vector"], 
        target_key="matrix",
        edge_type="forward"
    )
    
    circuit.add_mover(
        source_name="perch_1", 
        target_name="perch_2",
        source_keys=["matrix", "vector"], 
        target_key="matrix",
        edge_type="forward"
    )
    
    print(f"Added 3 movers connecting the perches")
    
    #--------------------------------------------------
    # 2. MODEL FINALIZATION
    #--------------------------------------------------
    print("\n2. MODEL FINALIZATION")
    
    # Define maps for the movers
    backward_map = {
        "operation": "square",
        "parameters": {}
    }
    
    forward_map = {
        "operation": "transform",
        "parameters": {
            "scale_factor": 0.1
        }
    }
    
    # Set maps for movers
    try:
        # Try set_mover_map first (older API)
        circuit.set_mover_map("perch_1", "perch_0", "backward", backward_map)
        circuit.set_mover_map("perch_0", "perch_1", "forward", forward_map)
        circuit.set_mover_map("perch_1", "perch_2", "forward", forward_map)
    except AttributeError:
        # If not available, use set_mover_comp or other alternative
        print("Using alternative API for setting mover operations...")
        # Define comp functions directly based on the maps
        def backward_comp(data):
            """Square each element in a vector"""
            vector = data.get("vector") if isinstance(data, dict) else data
            if vector is not None:
                return {"vector": vector**2}
            return {}
        
        def forward_comp_perch0_to_perch1(data):
            """Transform a matrix using a vector"""
            if isinstance(data, dict):
                matrix = data.get("matrix")
                vector = data.get("vector")
            else:
                matrix, vector = data if isinstance(data, tuple) and len(data) == 2 else (None, None)
            
            if matrix is not None and vector is not None:
                v_column = vector.reshape(-1, 1)
                outer_product = v_column @ v_column.T
                result_matrix = matrix + 0.1 * (matrix @ outer_product)
                return {"matrix": result_matrix}
            return {}
        
        def forward_comp_perch1_to_perch2(data):
            """Same transform function for perch1 to perch2"""
            return forward_comp_perch0_to_perch1(data)
        
        # Set the comp functions
        circuit.set_mover_comp("perch_1", "perch_0", "backward", backward_comp)
        circuit.set_mover_comp("perch_0", "perch_1", "forward", forward_comp_perch0_to_perch1)
        circuit.set_mover_comp("perch_1", "perch_2", "forward", forward_comp_perch1_to_perch2)
    
    # Finalize the model
    circuit.finalize_model()
    print(f"Circuit model finalized: has_model={circuit.has_model}")
    
    #--------------------------------------------------
    # 3. PORTABILITY
    #--------------------------------------------------
    print("\n3. PORTABILITY")
    
    # Define a comp factory
    def comp_factory(data):
        """Create a comp function from a map"""
        map_data = data.get("map", {})
        parameters = data.get("parameters", {})
        
        operation = map_data.get("operation")
        
        if operation == "square":
            def square_comp(data):
                """Square each element in a vector"""
                # Handle both dictionary inputs and direct numpy array inputs
                if isinstance(data, dict):
                    vector = data.get("vector")
                else:
                    # If data is directly the numpy array
                    vector = data
                    
                if vector is not None:
                    return {"vector": vector**2}
                return {}
            return square_comp
            
        elif operation == "transform":
            def transform_comp(data):
                """Transform a matrix using a vector"""
                # Handle both dictionary inputs and direct numpy array inputs
                if isinstance(data, dict):
                    matrix = data.get("matrix")
                    vector = data.get("vector")
                else:
                    # If we receive a tuple of (matrix, vector)
                    if isinstance(data, tuple) and len(data) == 2:
                        matrix, vector = data
                    else:
                        return {}
                
                if matrix is not None and vector is not None:
                    # Create a column vector
                    v_column = vector.reshape(-1, 1)
                    # Create an outer product (rank-1 update)
                    outer_product = v_column @ v_column.T
                    # Apply transformation: scale original matrix + rank-1 update
                    result_matrix = matrix + 0.1 * (matrix @ outer_product)
                    return {"matrix": result_matrix}
                return {}
            return transform_comp
            
        # Default case
        return lambda data: {}
    
    # Make the circuit portable
    circuit.make_portable(comp_factory)
    print(f"Circuit is portable: {circuit.is_portable}")
    
    #--------------------------------------------------
    # 4. INITIALIZATION
    #--------------------------------------------------
    print("\n4. INITIALIZATION")
    
    # Create initial values
    initial_vector = np.array([2.0, 3.0, 4.0])
    initial_matrix = np.array([
        [1.0, 0.1, 0.2],
        [0.1, 2.0, 0.3],
        [0.2, 0.3, 3.0]
    ])
    
    # Set initial values
    circuit.set_perch_data("perch_1", {"vector": initial_vector})
    circuit.set_perch_data("perch_0", {"matrix": initial_matrix})
    
    # Print lifecycle flags
    print("\nLIFECYCLE FLAGS:")
    print(f"has_empty_perches: {circuit.has_empty_perches}")
    print(f"has_model: {circuit.has_model}")
    print(f"movers_backward_exist: {circuit.movers_backward_exist}")
    print(f"is_portable: {circuit.is_portable}")
    print(f"is_solvable: {circuit.is_solvable}")
    print(f"is_solved: {circuit.is_solved}")
    print(f"is_simulated: {circuit.is_simulated}")
    
    #--------------------------------------------------
    # 5. INDIVIDUAL MOVER EXECUTION
    #--------------------------------------------------
    print("\n5. INDIVIDUAL MOVER EXECUTION")
    
    # Execute the backward mover from perch_1 to perch_0
    print("\nExecuting backward mover: perch_1 → perch_0")
    result = circuit.execute_mover("perch_1", "perch_0", edge_type="backward")
    print(f"Result: {result}")
    
    # Check perch_0 value after backward mover execution
    perch0_vector = circuit.get_perch_data("perch_0", "vector")
    print(f"perch_0 vector after backward mover: {perch0_vector}")
    
    # Execute the forward mover from perch_0 to perch_1
    print("\nExecuting forward mover: perch_0 → perch_1")
    result = circuit.execute_mover("perch_0", "perch_1", edge_type="forward")
    print(f"Result: {result}")
    
    # Check perch_1 matrix after forward mover execution
    perch1_matrix = circuit.get_perch_data("perch_1", "matrix")
    print(f"perch_1 matrix after forward mover:")
    print(perch1_matrix)
    
    # Execute the forward mover from perch_1 to perch_2
    print("\nExecuting forward mover: perch_1 → perch_2")
    result = circuit.execute_mover("perch_1", "perch_2", edge_type="forward")
    print(f"Result: {result}")
    
    # Check perch_2 matrix after forward mover execution
    perch2_matrix = circuit.get_perch_data("perch_2", "matrix")
    print(f"perch_2 matrix after forward mover:")
    print(perch2_matrix)
    
    # Check lifecycle flags after individual mover execution
    print("\nLIFECYCLE FLAGS AFTER INDIVIDUAL MOVER EXECUTION:")
    print(f"is_solved: {circuit.is_solved}")
    print(f"is_simulated: {circuit.is_simulated}")
    
    #--------------------------------------------------
    # 6. FULL SOLUTION
    #--------------------------------------------------
    print("\n6. FULL SOLUTION")
    
    # Reset perches to initial state for comparison
    circuit.set_perch_data("perch_0", {"vector": None, "matrix": initial_matrix})
    circuit.set_perch_data("perch_1", {"vector": initial_vector, "matrix": None})
    circuit.set_perch_data("perch_2", {"vector": None, "matrix": None})
    
    # Check if circuit is solvable
    if not circuit.is_solvable:
        print("Circuit is not solvable. Setting terminal values...")
        # In this example, we need to ensure perch_0 has vector value for backward solving
        # and perch_0 has matrix value for forward simulation
        circuit.set_perch_data("perch_0", {"vector": np.array([0.0, 0.0, 0.0])})
        print(f"Is circuit solvable now? {circuit.is_solvable}")
    
    try:
        # Solve the circuit in one step
        circuit.solve()
        print(f"Circuit solved: {circuit.is_solved} and simulated: {circuit.is_simulated}")
        
        # Get the results
        perch0_vector = circuit.get_perch_data("perch_0", "vector")
        perch1_matrix = circuit.get_perch_data("perch_1", "matrix")
        perch2_matrix = circuit.get_perch_data("perch_2", "matrix")
        
        # Print the results
        print("\nRESULTS AFTER FULL SOLUTION:")
        print(f"perch_0 vector: {perch0_vector}")
        print(f"perch_1 matrix shape: {perch1_matrix.shape if perch1_matrix is not None else 'None'}")
        print(f"perch_2 matrix shape: {perch2_matrix.shape if perch2_matrix is not None else 'None'}")
    except RuntimeError as e:
        print(f"Error solving circuit: {e}")
        print("Note: This circuit may not be solvable as a whole due to its structure.")
        print("Consider using individual mover execution as demonstrated earlier.")
    
    print("\nLifecycle flags at end of execution:")
    print(f"has_empty_perches: {circuit.has_empty_perches}")
    print(f"has_model: {circuit.has_model}")
    print(f"is_portable: {circuit.is_portable}")
    print(f"is_solvable: {circuit.is_solvable}")
    print(f"is_solved: {circuit.is_solved}")
    print(f"is_simulated: {circuit.is_simulated}")

if __name__ == "__main__":
    main()
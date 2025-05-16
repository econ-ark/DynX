import sys
import os
import unittest

# Define paths
modcraft_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
models_dir = os.path.join(modcraft_root, "models")
src_dir = os.path.join(modcraft_root, "src")

# Add paths to sys.path
if modcraft_root not in sys.path:
    sys.path.append(modcraft_root)
if models_dir not in sys.path:
    sys.path.append(models_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Print paths for debugging
print(f"Modcraft root: {modcraft_root}")
print(f"Models directory: {models_dir}")
print(f"Src directory: {src_dir}")
print(f"Python path: {sys.path}")

# Test imports
print("\nTesting imports:")
try:
    from dynx.stagecraft import Stage
    print("✓ Successfully imported Stage from dynx.stagecraft")
except ImportError as e:
    print(f"✗ Failed to import Stage from dynx.stagecraft: {e}")

try:
    from dynx.heptapodx import initialize_model
    print("✓ Successfully imported initialize_model from dynx.heptapodx")
except ImportError as e:
    print(f"✗ Failed to import initialize_model from dynx.heptapodx: {e}")

try:
    from DEGM.whisperer import operator_factory_in_situ
    print("✓ Successfully imported operator_factory_in_situ from DEGM.whisperer")
except ImportError as e:
    print(f"✗ Failed to import from DEGM.whisperer: {e}")

class TestImports(unittest.TestCase):
    """Test that we can import the main classes from the package."""
    
    def test_import_stage(self):
        """Test that we can import the Stage class."""
        try:
            from dynx.stagecraft import Stage
            self.assertTrue(True)  # If we get here, import succeeded
        except ImportError as e:
            self.fail(f"Failed to import Stage: {e}")

    def test_import_circuit_board(self):
        """Test that we can import the CircuitBoard class."""
        try:
            from dynx.core import CircuitBoard
            self.assertTrue(True)  # If we get here, import succeeded
        except ImportError as e:
            self.fail(f"Failed to import CircuitBoard: {e}")
            
    def test_import_perch(self):
        """Test that we can import the Perch class."""
        try:
            from dynx.core import Perch
            self.assertTrue(True)  # If we get here, import succeeded
        except ImportError as e:
            self.fail(f"Failed to import Perch: {e}")
            
    def test_import_mover(self):
        """Test that we can import the Mover class."""
        try:
            from dynx.core import Mover
            self.assertTrue(True)  # If we get here, import succeeded
        except ImportError as e:
            self.fail(f"Failed to import Mover: {e}")
            
    def test_import_period(self):
        """Test that we can import the Period class."""
        try:
            from dynx.stagecraft import Period
            self.assertTrue(True)  # If we get here, import succeeded
        except ImportError as e:
            self.fail(f"Failed to import Period: {e}")
            
    def test_import_model_circuit(self):
        """Test that we can import the ModelCircuit class."""
        try:
            from dynx.stagecraft import ModelCircuit
            self.assertTrue(True)  # If we get here, import succeeded
        except ImportError as e:
            self.fail(f"Failed to import ModelCircuit: {e}")

if __name__ == '__main__':
    unittest.main() 
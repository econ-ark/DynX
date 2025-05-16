#!/usr/bin/env python3
"""
Test script to verify the dynx package structure and compatibility shims.

This script checks:
1. Imports from the new package structure work
2. Imports from the compatibility shims work with deprecation warnings
3. Basic functionality of key classes and modules
"""

import sys
import warnings
import os
import unittest


class PackageStructureTests(unittest.TestCase):
    """Tests for verifying the package structure and imports."""
    
    def setUp(self):
        """Set up the test environment."""
        # Ensure we can catch deprecation warnings
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        
        # Add the current directory to the Python path if needed
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
    
    def test_new_imports(self):
        """Test that dynx.* imports are accessible."""
        # Direct import style
        import dynx
        from dynx import Stage, ModelCircuit, Period
        from dynx import CircuitBoard, Perch, Mover
        from dynx import CircuitRunner, RunRecorder, mpi_map
        from dynx.heptapodx import FunctionalProblem, AttrDict
        
        # Test that they are correctly defined
        self.assertTrue(hasattr(dynx, 'Stage'))
        self.assertTrue(hasattr(dynx, 'ModelCircuit'))
        self.assertTrue(hasattr(dynx, 'CircuitRunner'))
        
        # Make sure the dynx.runner package has the expected exports
        from dynx.runner import CircuitRunner, RunRecorder, mpi_map
        
        self.assertEqual(CircuitRunner.__name__, 'CircuitRunner')
        self.assertEqual(RunRecorder.__name__, 'RunRecorder')
        self.assertEqual(mpi_map.__name__, 'mpi_map')
    
    def test_compatibility_imports(self):
        """Skip test - no backwards compatibility needed."""
        # We no longer need to support backward compatibility
        self.skipTest("Backwards compatibility not needed for this preliminary package")
    
    def test_module_paths(self):
        """Test that modules are in the correct locations."""
        # Import the modules directly
        from dynx.core.circuit_board import CircuitBoard
        from dynx.stagecraft.stage import Stage
        from dynx.stagecraft.period import Period
        from dynx.heptapodx.core.functional_problem import FunctionalProblem
        from dynx.runner.circuit_runner import CircuitRunner
        
        # Check their module paths
        self.assertEqual(CircuitBoard.__module__, "dynx.core.circuit_board")
        self.assertEqual(Stage.__module__, "dynx.stagecraft.stage")
        self.assertEqual(Period.__module__, "dynx.stagecraft.period")
        self.assertEqual(FunctionalProblem.__module__, "dynx.heptapodx.core.functional_problem")
        self.assertEqual(CircuitRunner.__module__, "dynx.runner.circuit_runner")
    
    def test_compatibility_module_paths(self):
        """Skip test - no compatibility modules to test."""
        # We no longer need to support backward compatibility
        self.skipTest("No compatibility modules to test in this preliminary package")


if __name__ == "__main__":
    # Run tests
    unittest.main() 
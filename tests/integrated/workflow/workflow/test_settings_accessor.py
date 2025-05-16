#!/usr/bin/env python3
"""
Test to verify the settings accessor in FunctionalProblem class.
"""

import os
import sys
from pathlib import Path

# Add the repository root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, repo_root)

from src.heptapod_b.core.functional_problem import FunctionalProblem

def test_settings_accessor():
    # Create a FunctionalProblem instance
    problem = FunctionalProblem()
    
    # Add some test settings
    problem.settings_dict = {
        'tolerance': 1e-6,
        'max_iterations': 100,
        'algorithm': 'newton'
    }
    
    # Test dictionary access
    assert problem.settings_dict['tolerance'] == 1e-6
    assert problem.settings_dict['max_iterations'] == 100
    assert problem.settings_dict['algorithm'] == 'newton'
    
    # Test property accessor access
    assert problem.settings.tolerance == 1e-6
    assert problem.settings.max_iterations == 100
    assert problem.settings.algorithm == 'newton'
    
    # Test dir() for completion
    assert set(['tolerance', 'max_iterations', 'algorithm']) == set(dir(problem.settings))
    
    print("✅ Settings accessor tests passed!")
    return True

if __name__ == "__main__":
    test_settings_accessor() 
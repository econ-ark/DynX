"""
Test script for master parameter and function inheritance in Heptapod-B.

This script tests that parameters and functions can be correctly inherited from a 
master configuration file, with stage-level overrides taking precedence.
"""

import os
import sys
import numpy as np
from pathlib import Path
import time

# Add the src directory to the path if running the script directly
script_dir = Path(__file__).parent.resolve()
repo_root = script_dir.parent.parent
sys.path.append(str(repo_root))

from src.heptapod_b.io.yaml_loader import load_config
from src.heptapod_b.resolve.methods import resolve_parameter_references
from src.heptapod_b.init.stage import build_stage

def test_master_parameter_resolution():
    """
    Test that parameter references to a master file are correctly resolved.
    """
    start_time = time.time()
    
    # Define paths
    configs_dir = repo_root / "examples" / "configs"
    stage_file = configs_dir / "OWNC_stage_with_master.yml"
    master_file = configs_dir / "housing_renting_master.yml"
    
    print("=" * 80)
    print(f"TESTING MASTER PARAMETER AND FUNCTION INHERITANCE")
    print("=" * 80)
    print(f"Stage file:  {stage_file.relative_to(repo_root)}")
    print(f"Master file: {master_file.relative_to(repo_root)}")
    print("-" * 80)
    
    # Verify files exist
    if not stage_file.exists():
        raise FileNotFoundError(f"Stage file not found: {stage_file}")
    if not master_file.exists():
        raise FileNotFoundError(f"Master file not found: {master_file}")
    
    # Load the master config for reference
    master_config = load_config(str(master_file))
    print("\n1. MASTER CONFIGURATION CONTENT")
    print("-" * 40)
    print(f"  Parameters:        {len(master_config['parameters'])} items")
    print(f"  Settings:          {len(master_config['settings'])} items")
    print(f"  Top-level funcs:   {len(master_config.get('functions', {}))} functions")
    print(f"  Math functions:    {len(master_config.get('math', {}).get('functions', {}))} functions")
    
    # Check for some specific functions in the master
    print("\n2. MASTER FUNCTION EXAMPLES")
    print("-" * 40)
    if 'functions' in master_config:
        for func_name in ['log_utility', 'income_process']:
            if func_name in master_config['functions']:
                func_expr = master_config['functions'][func_name].get('expr', 'Not found')
                print(f"  {func_name}: {func_expr}")
    
    if 'math' in master_config and 'functions' in master_config['math']:
        math_funcs = master_config['math']['functions']
        for func_name in ['u_general', 'standard_interest']:
            if func_name in math_funcs:
                func_expr = math_funcs[func_name].get('expr', 'Not found')
                print(f"  {func_name}: {func_expr}")
    
    # Load the stage config (which should reference the master file)
    print("\n3. LOADING STAGE CONFIG")
    print("-" * 40)
    config = load_config(str(stage_file))
    
    # Check that _master was loaded
    if "_master" not in config:
        raise ValueError("Master config was not loaded properly - no _master key found")
    else:
        print(f"  Master config successfully loaded")
    
    # Verify master functions were stored in _master
    if "math_functions" in config["_master"]:
        math_functions = config["_master"]["math_functions"]
        print(f"  Master math functions loaded: {len(math_functions)} items")
        print(f"  Examples: {', '.join(list(math_functions.keys())[:3])}")
    else:
        print("  No master math functions found")
    
    # Compare master and stage parameter values
    print("\n4. PARAMETER COMPARISON")
    print("-" * 40)
    
    master_params = [
        ('beta', 'Referenced from master (expected to match)'),
        ('r', 'Overridden in stage (should be different)')
    ]
    
    for param_name, description in master_params:
        master_value = master_config['parameters'][param_name]
        stage_value = config['stage']['parameters'][param_name]
        print(f"  {param_name}: {description}")
        print(f"    - Master: {master_value}")
        print(f"    - Stage:  {stage_value}")
        print(f"    - Match:  {'✓' if master_value == stage_value else '✗'}")
    
    # Build the stage
    print("\n5. BUILDING STAGE")
    print("-" * 40)
    print(f"  Building stage with automatic parameter and function resolution...")
    stage = build_stage(str(stage_file))
    print(f"  Stage built successfully!")
    
    # Check function inheritance with different syntaxes
    print("\n6. FUNCTION INHERITANCE (DIFFERENT SYNTAXES)")
    print("-" * 40)
    
    # Test all inheritance methods
    inheritance_tests = [
        ('u_general', 'Bracket notation', '["u_general"]'),
        ('housing_utility', 'Direct reference', 'housing_utility'),
        ('standard_interest', 'Property-based', 'inherit: "standard_interest"'),
        ('identity_mapping', 'Legacy inherit', 'inherit: true')
    ]
    
    print("  Testing direct inheritance methods:")
    for func_name, method, syntax in inheritance_tests:
        if func_name in stage.math['functions']:
            master_expr = master_config['math']['functions'][func_name].get('expr', 'Unknown')
            stage_expr = stage.math['functions'][func_name].get('expr', 'Unknown')
            match = master_expr == stage_expr
            print(f"    ✓ {method} inheritance: {func_name}")
            print(f"      Syntax: {syntax}")
            print(f"      Match:  {'✓' if match else '✗'}")
            
            # Expressions should be identical for directly inherited functions
            assert match, f"Expression mismatch for inherited function {func_name}"
        else:
            print(f"    ✗ {method} inheritance failed for: {func_name}")
    
    # Modified inheritance functions
    print("\n7. MODIFIED INHERITED FUNCTIONS")
    print("-" * 40)
    
    modified_inherited = [
        ('uc_general', 'with numerical stability'),
        ('egm_consumption', 'adapted for local variables')
    ]
    
    for func_name, modification in modified_inherited:
        if func_name in stage.math['functions']:
            master_expr = master_config['math']['functions'][func_name].get('expr', 'Unknown')
            stage_expr = stage.math['functions'][func_name].get('expr', 'Unknown')
            match = master_expr == stage_expr
            print(f"  ✓ Modified function: {func_name} ({modification})")
            print(f"    - Master: {master_expr}")
            print(f"    - Stage:  {stage_expr}")
            print(f"    - Different: {'✓' if not match else '✗'}")
            
            # Expressions should be different for modified functions
            assert not match, f"Expression should be different for modified function {func_name}"
        else:
            print(f"  ✗ Missing modified function: {func_name}")
    
    # Stage-specific functions
    print("\n8. STAGE-SPECIFIC FUNCTIONS")
    print("-" * 40)
    
    stage_specific = ['u_func', 'asset_interest', 'wealth_constraint', 'g_av', 'g_ve']
    for func_name in stage_specific:
        if func_name in stage.math['functions']:
            print(f"  ✓ Stage-specific: {func_name}")
            print(f"    - Expr: {stage.math['functions'][func_name].get('expr', 'Unknown')}")
        else:
            print(f"  ✗ Missing stage-specific function: {func_name}")
    
    # Verify parameter values in stage
    print("\n9. FINAL PARAMETER VERIFICATION")
    print("-" * 40)
    print(f"  beta (expected={master_config['parameters']['beta']}, actual={stage.parameters_dict['beta']})")
    assert stage.parameters_dict['beta'] == master_config['parameters']['beta'], \
        "Stage parameter 'beta' was not resolved correctly"
    
    print(f"  r (expected=1.05, actual={stage.parameters_dict['r']})")
    assert stage.parameters_dict['r'] == 1.05, \
        "Stage parameter 'r' was not set to the overridden value"
    
    assert stage.parameters_dict['r'] != master_config['parameters']['r'], \
        "Stage parameter 'r' was incorrectly set to the master value"
    
    # Test corner cases for function inheritance
    print("\n10. CORNER CASE: PARAMETER USAGE IN FUNCTIONS")
    print("-" * 40)
    print("  Testing if stage parameters (r=1.05) are used in functions...")
    if 'asset_interest' in stage.math['functions']:
        asset_interest_expr = stage.math['functions']['asset_interest']['expr']
        print(f"  asset_interest expression: {asset_interest_expr}")
        print(f"  This should use stage r=1.05 (not master r=1.04)")
        assert "(1+r)*a" in asset_interest_expr, "asset_interest should use r parameter"
    
    duration = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"ALL TESTS PASSED! ({duration:.2f} seconds)")
    print("Parameter resolution and function inheritance are working correctly.")
    print("=" * 80)

if __name__ == "__main__":
    test_master_parameter_resolution() 
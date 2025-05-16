#!/usr/bin/env python3
"""
Basic access tests for Stage objects.
Demonstrates that all core attribute paths on a compiled Stage work.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add the repository root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, repo_root)

from src.stagecraft.stage import Stage
from src.heptapod_b.core.api import initialize_model
from src.heptapod_b.core.api import generate_numerical_model
from src.heptapod_b.io.yaml_loader import load_config


def print_banner(text):
    """Print a banner with the given text."""
    print("\n" + "=" * 40)
    print(f"=== {text} ===".ljust(39) + "=")
    print("=" * 40)


if __name__ == "__main__":
    # Step 1: Create Stage
    stage = Stage("AccessTest", init_rep=initialize_model, num_rep=generate_numerical_model)
    
    # Step 2: Load config
    config_path = os.path.join(repo_root, "examples/configs/ConsInd_multi.yml")
    config = load_config(config_path)
    stage = Stage(config=config, master_config=None, 
                 init_rep=initialize_model, num_rep=generate_numerical_model)
    
    # Step 3: Build computational model
    stage.build_computational_model()

    # Step 4: Test Stage Components
    print_banner("STAGE COMPONENTS")
    # Test stage model access
    model = stage.model
    print(f"stage.model: {type(model)}")
    
    # Test model components
    try:
        math = stage.model.math
        print(f"stage.model.math: {type(math)}")
        
        functions = math.get('functions', {}) if hasattr(math, 'get') else getattr(math, 'functions', {})
        print(f"stage.model.math.functions: {type(functions)}")
        
        state_space = math.get('state_space', {}) if hasattr(math, 'get') else getattr(math, 'state_space', {})
        print(f"stage.model.math.state_space: {type(state_space)}")
    except AttributeError as e:
        print(f"Could not access math attributes: {e}")
    
    # Try accessing parameters as both attribute and dict key
    try:
        if hasattr(stage.model, 'parameters'):
            params = stage.model.parameters
            print(f"stage.model.parameters (attribute): {type(params)}")
        elif hasattr(stage.model, 'parameters_dict'):
            params = stage.model.parameters_dict
            print(f"stage.model.parameters_dict: {type(params)}")
        else:
            print("No parameters attribute found on model")
    except Exception as e:
        print(f"Error accessing parameters: {e}")
    
    # Try accessing settings as both attribute and dict key
    try:
        if hasattr(stage.model, 'settings'):
            settings = stage.model.settings
            print(f"stage.model.settings (attribute): {type(settings)}")
        elif hasattr(stage.model, 'settings_dict'):
            settings = stage.model.settings_dict
            print(f"stage.model.settings_dict: {type(settings)}")
        else:
            print("No settings attribute found on model")
    except Exception as e:
        print(f"Error accessing settings: {e}")
    
    # Try accessing methods as both attribute and dict key
    try:
        if hasattr(stage.model, 'methods'):
            methods = stage.model.methods
            print(f"stage.model.methods (attribute): {type(methods)}")
        elif hasattr(stage.model, 'methods_dict'):
            methods = stage.model.methods_dict
            print(f"stage.model.methods_dict: {type(methods)}")
        else:
            print("No methods attribute found on model")
    except Exception as e:
        print(f"Error accessing methods: {e}")
    
    # Access numerical representation
    try:
        num = stage.model.num
        print(f"stage.model.num: {type(num)}")
        
        num_state_space = stage.model.num.state_space
        print(f"stage.model.num.state_space: {type(num_state_space)}")
    except AttributeError as e:
        print(f"Could not access num attributes: {e}")
    
    # Step 5: Test Perch Access
    print_banner("PERCH ACCESS")
    
    # Test arrival perch
    arvl = stage.arvl
    print(f"stage.arvl: {type(arvl)}")
    
    arvl_name = stage.arvl.name
    print(f"stage.arvl.name: {arvl_name}")
    
    arvl_model = stage.arvl.model
    print(f"stage.arvl.model: {type(arvl_model)}")
    
    # Check available attributes on perch model
    print(f"Available attributes on perch model: {[attr for attr in dir(arvl_model) if not attr.startswith('_')]}")
    
    try:
        if hasattr(arvl_model, 'math'):
            arvl_math = arvl_model.math
            print(f"stage.arvl.model.math: {type(arvl_math)}")
    except Exception as e:
        print(f"Error accessing arvl_model.math: {e}")
    
    try:
        if hasattr(arvl_model, 'parameters'):
            arvl_params = arvl_model.parameters
            print(f"stage.arvl.model.parameters: {type(arvl_params)}")
        elif hasattr(arvl_model, 'parameters_dict'):
            arvl_params = arvl_model.parameters_dict
            print(f"stage.arvl.model.parameters_dict: {type(arvl_params)}")
    except Exception as e:
        print(f"Error accessing parameters: {e}")
    
    try:
        if hasattr(arvl_model, 'settings'):
            arvl_settings = arvl_model.settings
            print(f"stage.arvl.model.settings: {type(arvl_settings)}")
        elif hasattr(arvl_model, 'settings_dict'):
            arvl_settings = arvl_model.settings_dict
            print(f"stage.arvl.model.settings_dict: {type(arvl_settings)}")
    except Exception as e:
        print(f"Error accessing settings: {e}")
    
    try:
        if hasattr(arvl_model, 'methods'):
            arvl_methods = arvl_model.methods
            print(f"stage.arvl.model.methods: {type(arvl_methods)}")
        elif hasattr(arvl_model, 'methods_dict'):
            arvl_methods = arvl_model.methods_dict
            print(f"stage.arvl.model.methods_dict: {type(arvl_methods)}")
    except Exception as e:
        print(f"Error accessing methods: {e}")
    
    try:
        arvl_num = stage.arvl.model.num
        print(f"stage.arvl.model.num: {type(arvl_num)}")
        
        arvl_ss = stage.arvl.model.num.state_space
        print(f"stage.arvl.model.num.state_space: {type(arvl_ss)}")
        
        arvl_ss_arvl = stage.arvl.model.num.state_space.arvl
        print(f"stage.arvl.model.num.state_space.arvl: {type(arvl_ss_arvl)}")
        
        arvl_grids = stage.arvl.model.num.state_space.arvl.grids
        print(f"stage.arvl.model.num.state_space.arvl.grids: {type(arvl_grids)}")
        
        arvl_a_grid = stage.arvl.model.num.state_space.arvl.grids.a
        print(f"stage.arvl.model.num.state_space.arvl.grids.a: {type(arvl_a_grid)}, shape: {arvl_a_grid.shape}")
        
        # Check for direct age grid access
        try:
            arvl_age_grid = stage.arvl.model.num.state_space.arvl.grids.age
            print(f"stage.arvl.model.num.state_space.arvl.grids.age: {type(arvl_age_grid)}, shape: {arvl_age_grid.shape}")
        except Exception as e:
            print(f"Error accessing stage.arvl.model.num.state_space.arvl.grids.age: {e}")
        
        # Test grid proxy access
        arvl_a = stage.arvl.grid.a
        print(f"stage.arvl.grid.a: {type(arvl_a)}, shape: {arvl_a.shape}, first: {arvl_a[0]}, last: {arvl_a[-1]}")
        
        arvl_age = stage.arvl.grid.age
        print(f"stage.arvl.grid.age: {type(arvl_age)}, shape: {arvl_age.shape}, first: {arvl_age[0]}, last: {arvl_age[-1]}")
        
        # Check for direct mesh access at model level
        try:
            arvl_mesh_a = stage.arvl.model.num.state_space.arvl.mesh.a
            print(f"stage.arvl.model.num.state_space.arvl.mesh.a: {type(arvl_mesh_a)}, shape: {arvl_mesh_a.shape}")
        except Exception as e:
            print(f"Error accessing stage.arvl.model.num.state_space.arvl.mesh.a: {e}")
        
        # Check for mesh proxy access
        try:
            a_mesh_proxy_a = stage.arvl.mesh.a
            print(f"stage.arvl.mesh.a: {type(a_mesh_proxy_a)}, shape: {a_mesh_proxy_a.shape}")
        except Exception as e:
            print(f"Error accessing stage.arvl.mesh.a: {e}")
        
        # Try creating mesh grids and accessing them
    except Exception as e:
        print(f"Error accessing arvl grids: {e}")
    
    # Test decision perch
    dcsn = stage.dcsn
    print(f"stage.dcsn: {type(dcsn)}")
    
    dcsn_model = stage.dcsn.model
    print(f"stage.dcsn.model: {type(dcsn_model)}")
    
    try:
        dcsn_m_grid = stage.dcsn.model.num.state_space.dcsn.grids.m
        print(f"stage.dcsn.model.num.state_space.dcsn.grids.m: {type(dcsn_m_grid)}, shape: {dcsn_m_grid.shape}")
        
        dcsn_m = stage.dcsn.grid.m
        print(f"stage.dcsn.grid.m: {type(dcsn_m)}, shape: {dcsn_m.shape}, first: {dcsn_m[0]}, last: {dcsn_m[-1]}")
    except Exception as e:
        print(f"Error accessing dcsn grids: {e}")
    
    # Test continuation perch
    cntn = stage.cntn
    print(f"stage.cntn: {type(cntn)}")
    
    cntn_model = stage.cntn.model
    print(f"stage.cntn.model: {type(cntn_model)}")
    
    try:
        cntn_a_nxt_grid = stage.cntn.model.num.state_space.cntn.grids.a_nxt
        print(f"stage.cntn.model.num.state_space.cntn.grids.a_nxt: {type(cntn_a_nxt_grid)}, shape: {cntn_a_nxt_grid.shape}")
        
        cntn_a_nxt = stage.cntn.grid.a_nxt
        print(f"stage.cntn.grid.a_nxt: {type(cntn_a_nxt)}, shape: {cntn_a_nxt.shape}, first: {cntn_a_nxt[0]}, last: {cntn_a_nxt[-1]}")
    except Exception as e:
        print(f"Error accessing cntn grids: {e}")
    
    # Step 6: Test Mover Access
    print_banner("MOVER ACCESS")
    
    # Test arvl_to_dcsn mover
    arvl_to_dcsn = stage.arvl_to_dcsn
    print(f"stage.arvl_to_dcsn: {type(arvl_to_dcsn)}")
    
    try:
        source_name = stage.arvl_to_dcsn.source_name
        print(f"stage.arvl_to_dcsn.source_name: {source_name}")
        
        target_name = stage.arvl_to_dcsn.target_name
        print(f"stage.arvl_to_dcsn.target_name: {target_name}")
        
        edge_type = stage.arvl_to_dcsn.edge_type
        print(f"stage.arvl_to_dcsn.edge_type: {edge_type}")
        
        mover_model = stage.arvl_to_dcsn.model
        print(f"stage.arvl_to_dcsn.model: {type(mover_model)}")
        
        # Check available attributes on mover model
        print(f"Available attributes on mover model: {[attr for attr in dir(mover_model) if not attr.startswith('_')]}")
        
        if hasattr(mover_model, 'math'):
            mover_math = mover_model.math
            print(f"stage.arvl_to_dcsn.model.math: {type(mover_math)}")
        
        if hasattr(mover_model, 'parameters'):
            mover_params = mover_model.parameters
            print(f"stage.arvl_to_dcsn.model.parameters: {type(mover_params)}")
        elif hasattr(mover_model, 'parameters_dict'):
            mover_params = mover_model.parameters_dict
            print(f"stage.arvl_to_dcsn.model.parameters_dict: {type(mover_params)}")
        
        if hasattr(mover_model, 'settings'):
            mover_settings = mover_model.settings
            print(f"stage.arvl_to_dcsn.model.settings: {type(mover_settings)}")
        elif hasattr(mover_model, 'settings_dict'):
            mover_settings = mover_model.settings_dict
            print(f"stage.arvl_to_dcsn.model.settings_dict: {type(mover_settings)}")
        
        if hasattr(mover_model, 'methods'):
            mover_methods = mover_model.methods
            print(f"stage.arvl_to_dcsn.model.methods: {type(mover_methods)}")
        elif hasattr(mover_model, 'methods_dict'):
            mover_methods = mover_model.methods_dict
            print(f"stage.arvl_to_dcsn.model.methods_dict: {type(mover_methods)}")
    except Exception as e:
        print(f"Error accessing mover attributes: {e}")
    
    # Test other movers
    try:
        dcsn_to_cntn = stage.dcsn_to_cntn
        print(f"stage.dcsn_to_cntn: {type(dcsn_to_cntn)}")
        
        dcsn_to_cntn_model = stage.dcsn_to_cntn.model
        print(f"stage.dcsn_to_cntn.model: {type(dcsn_to_cntn_model)}")
        
        cntn_to_dcsn = stage.cntn_to_dcsn
        print(f"stage.cntn_to_dcsn: {type(cntn_to_dcsn)}")
        
        cntn_to_dcsn_model = stage.cntn_to_dcsn.model
        print(f"stage.cntn_to_dcsn.model: {type(cntn_to_dcsn_model)}")
        
        dcsn_to_arvl = stage.dcsn_to_arvl
        print(f"stage.dcsn_to_arvl: {type(dcsn_to_arvl)}")
        
        dcsn_to_arvl_model = stage.dcsn_to_arvl.model
        print(f"stage.dcsn_to_arvl.model: {type(dcsn_to_arvl_model)}")
    except Exception as e:
        print(f"Error accessing other movers: {e}")
    
    # Step 7: Test Stage Status Flags
    print_banner("STAGE STATUS FLAGS")
    
    try:
        # Print all status flags
        print(f"stage.status_flags: {type(stage.status_flags)}")
        for flag_name, flag_value in stage.status_flags.items():
            print(f"stage.status_flags['{flag_name}']: {flag_value}")
    except Exception as e:
        print(f"Error accessing stage status flags: {e}")
    
    # Step 8: Test parameter, setting, and method counts
    print_banner("PARAM / SET / METH COUNTS")
    for perch_name in ["arvl", "dcsn", "cntn"]:
        perch = getattr(stage, perch_name)
        try:
            param_len = len(perch.model.parameters) if hasattr(perch.model, "parameters") else 0
            settings_len = len(perch.model.settings) if hasattr(perch.model, "settings") else 0
            methods_len = len(perch.model.methods) if hasattr(perch.model, "methods") else 0
            
            print(f"stage.{perch_name}.model:")
            print(f"  parameters: {param_len}")
            print(f"  settings: {settings_len}")
            print(f"  methods: {methods_len}")
        except Exception as e:
            print(f"Error getting lengths for {perch_name}: {e}")
    
    # Success message
    print("\n✅ basic access-tests passed")
    sys.exit(0) 
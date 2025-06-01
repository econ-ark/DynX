#!/usr/bin/env python
"""
Example demonstrating Solution object re-hydration when loading saved models.

This example shows how the io module automatically converts Solution dictionaries
back to proper Solution objects when loading, ensuring compatibility with 
downstream code that expects Solution attributes.
"""

import sys
import os
from pathlib import Path

# Add codebase to path if running directly
if __name__ == "__main__":
    codebase_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(codebase_dir))

import numpy as np
from dynx.stagecraft.solmaker import Solution

def demonstrate_solution_behavior():
    """Show how Solution objects behave with dict conversion."""
    print("=== Solution Object Behavior ===\n")
    
    # Create a Solution object
    sol = Solution()
    sol.policy["c"] = np.array([1.0, 2.0, 3.0])
    sol.policy["H"] = np.array([0.5, 0.6, 0.7])
    sol.EGM.unrefined.e = np.array([0.1, 0.2, 0.3])
    sol.EGM.refined.e = np.array([0.11, 0.21, 0.31])
    sol.timing["solve_time"] = 1.23
    sol.timing["iterations"] = 10
    
    print("Original Solution object:")
    print(f"  Type: {type(sol)}")
    print(f"  Can access sol.policy['c']: {sol.policy['c']}")
    print(f"  Can access sol.EGM.unrefined.e: {sol.EGM.unrefined.e}")
    
    # Convert to dict (what happens during save)
    sol_dict = sol.as_dict()
    print("\nAfter as_dict() conversion:")
    print(f"  Type: {type(sol_dict)}")
    print(f"  Keys: {list(sol_dict.keys())}")
    
    # Show what happens without re-hydration
    print("\nWithout re-hydration (plain dict):")
    try:
        # This would fail with a dict
        _ = sol_dict.EGM.unrefined.e
    except AttributeError as e:
        print(f"  ERROR: {e}")
        print("  Dict access would need: sol_dict['EGM']['unrefined']['e']")
    
    # Show re-hydration
    sol_rehydrated = Solution.from_dict(sol_dict)
    print("\nAfter re-hydration:")
    print(f"  Type: {type(sol_rehydrated)}")
    print(f"  Can access sol.policy['c']: {sol_rehydrated.policy['c']}")
    print(f"  Can access sol.EGM.unrefined.e: {sol_rehydrated.EGM.unrefined.e}")

def explain_automatic_rehydration():
    """Explain how automatic re-hydration works in load_circuit."""
    print("\n=== Automatic Re-hydration in load_circuit ===\n")
    
    print("When load_circuit loads a saved model:")
    print("1. It unpickles the .pkl files")
    print("2. If the object is a dict with keys {'policy', 'EGM', 'timing'}")
    print("   → It's recognized as a Solution and converted using Solution.from_dict()")
    print("3. Otherwise, the object is used as-is")
    print()
    print("This ensures that downstream code expecting Solution attributes")
    print("(like plot_policy(stage.dcsn.sol.EGM.unrefined.e)) works correctly.")
    print()
    print("Detection logic:")
    print('  if isinstance(obj, dict) and {"policy", "EGM", "timing"} <= obj.keys():')
    print('      obj = Solution.from_dict(obj)')

def main():
    """Run all demonstrations."""
    demonstrate_solution_behavior()
    explain_automatic_rehydration()
    
    print("\n=== Benefits ===")
    print("• Backward compatibility: Old saved models are automatically fixed")
    print("• Forward compatibility: New Solution formats work seamlessly")
    print("• No manual intervention needed when loading models")
    print("• Plotting and analysis code continues to work as expected")

if __name__ == "__main__":
    main() 
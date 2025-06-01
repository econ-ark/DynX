#!/usr/bin/env python
"""
Example demonstrating branch dictionary handling for stages with multiple solution branches.

This example shows how the io module handles cases like TENU.cntn.sol where the solution
is not a single Solution object but a dictionary mapping branch keys to Solution objects:
{"from_owner": <Solution>, "from_renter": <Solution>}
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

def create_branch_solutions():
    """Create example branch solutions like those in TENU stage."""
    print("=== Creating Branch Solutions ===\n")
    
    # Create owner branch solution
    owner_sol = Solution()
    owner_sol.policy["c"] = np.array([10.0, 11.0, 12.0])
    owner_sol.policy["H"] = np.array([0.8, 0.85, 0.9])
    owner_sol.EGM.unrefined.e = np.array([1.0, 1.1, 1.2])
    owner_sol.EGM.refined.e = np.array([1.01, 1.11, 1.21])
    owner_sol.timing["solve_time"] = 0.5
    owner_sol.timing["iterations"] = 5.0  # float for consistency
    
    # Create renter branch solution
    renter_sol = Solution()
    renter_sol.policy["c"] = np.array([8.0, 9.0, 10.0])
    renter_sol.policy["H"] = np.array([0.0, 0.0, 0.0])  # Renters don't own housing
    renter_sol.EGM.unrefined.e = np.array([0.8, 0.9, 1.0])
    renter_sol.EGM.refined.e = np.array([0.81, 0.91, 1.01])
    renter_sol.timing["solve_time"] = 0.3
    renter_sol.timing["iterations"] = 3.0  # float for consistency
    
    # Create branch dictionary
    branch_dict = {
        "from_owner": owner_sol,
        "from_renter": renter_sol
    }
    
    print("Created branch dictionary with keys:", list(branch_dict.keys()))
    print(f"  from_owner: {type(branch_dict['from_owner'])}")
    print(f"  from_renter: {type(branch_dict['from_renter'])}")
    
    return branch_dict

def demonstrate_serialization_problem():
    """Show what happens without proper branch dict handling."""
    print("\n=== The Problem: Numba typed.Dict Not Pickleable ===\n")
    
    print("Without special handling:")
    print("1. _dump_object tries to pickle the branch dict directly")
    print("2. Each Solution contains numba typed.Dict objects (for policy, etc.)")
    print("3. pickle.dump raises: TypeError: can't pickle numba.typed.Dict")
    print("4. Exception handler logs 'not pickle-able' and leaves empty sol.pkl")
    print("5. Loading fails with EOFError on empty file")
    print("6. Plotting code can't find sol['from_owner'].EGM.unrefined.e")

def explain_solution():
    """Explain how the fix works."""
    print("\n=== The Solution: Smart Branch Dict Handling ===\n")
    
    print("In _dump_object:")
    print("1. Detect dicts containing Solution objects")
    print("2. Convert each Solution to plain dict using as_dict()")
    print("3. Pickle the resulting dict-of-dicts")
    print()
    print("In load_circuit:")
    print("1. After unpickling, detect branch dictionaries")
    print("2. For each branch that looks like a Solution dict")
    print("3. Re-hydrate it using Solution.from_dict()")
    print()
    print("Result: Branch solutions are preserved and usable!")

def show_usage_example():
    """Show how to access branch solutions in practice."""
    print("\n=== Accessing Branch Solutions ===\n")
    
    branch_dict = create_branch_solutions()
    
    print("Example usage after loading:")
    print("  # Access owner branch")
    print("  owner_policy = stage.cntn.sol['from_owner'].policy['c']")
    print(f"  Result: {branch_dict['from_owner'].policy['c']}")
    print()
    print("  # Access renter branch EGM")
    print("  renter_egm = stage.cntn.sol['from_renter'].EGM.unrefined.e")
    print(f"  Result: {branch_dict['from_renter'].EGM.unrefined.e}")
    print()
    print("  # Check housing ownership")
    print("  owner_H = stage.cntn.sol['from_owner'].policy['H']")
    print("  renter_H = stage.cntn.sol['from_renter'].policy['H']")
    print(f"  Owner housing: {branch_dict['from_owner'].policy['H']}")
    print(f"  Renter housing: {branch_dict['from_renter'].policy['H']}")

def main():
    """Run all demonstrations."""
    branch_dict = create_branch_solutions()
    demonstrate_serialization_problem()
    explain_solution()
    show_usage_example()
    
    print("\n=== Benefits ===")
    print("• TENU stage solutions save and load correctly")
    print("• No more zero-byte sol.pkl files")
    print("• No more EOFError when loading")
    print("• Plotting code finds branch solutions as expected")
    print("• Works for any stage with branch-based solutions")

if __name__ == "__main__":
    main() 
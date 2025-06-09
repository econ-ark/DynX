"""
Solution container for FUES/StageCraft models.

This module provides a Numba-compatible container for storing stage solutions
with support for both attribute and dictionary-style access.
"""

import numpy as np
import pickle
import json
from types import SimpleNamespace
from collections.abc import MutableMapping
from numba import njit, typed, types
from numba.core.types import DictType


class _AttrView(MutableMapping):
    """Wrapper that provides attribute-style access to a typed.Dict."""
    
    def __init__(self, typed_dict, shapes_dict=None):
        self._dict = typed_dict
        self._shapes = shapes_dict or {}
    
    def __getattr__(self, key):
        if key.startswith('_'):
            return object.__getattribute__(self, key)
        try:
            arr = self._dict[key]
            # Reshape if we have shape info
            if key in self._shapes:
                return arr.reshape(self._shapes[key])
            return arr
        except KeyError:
            raise AttributeError(f"No attribute '{key}'")
    
    def __setattr__(self, key, value):
        if key.startswith('_'):
            object.__setattr__(self, key, value)
        else:
            if isinstance(value, np.ndarray):
                # Store shape and flatten
                self._shapes[key] = value.shape
                self._dict[key] = value.flatten()
            else:
                self._dict[key] = value
    
    def __getitem__(self, key):
        arr = self._dict[key]
        # Reshape if we have shape info
        if key in self._shapes:
            return arr.reshape(self._shapes[key])
        return arr
    
    def __setitem__(self, key, value):
        if isinstance(value, np.ndarray):
            # Store shape and flatten
            self._shapes[key] = value.shape
            self._dict[key] = value.flatten()
        else:
            self._dict[key] = value
    
    def __delitem__(self, key):
        del self._dict[key]
        if key in self._shapes:
            del self._shapes[key]
    
    def __iter__(self):
        return iter(self._dict)
    
    def __len__(self):
        return len(self._dict)
    
    def __repr__(self):
        return f"_AttrView({dict(self._dict)})"


class Solution:
    """
    Numba-compatible solution container for FUES/StageCraft models.
    
    Provides storage for core solution arrays (policy, vlu, Q, lambda_, phi)
    and nested dictionaries (timing, EGM layers) with both attribute and 
    dictionary-style access.
    
    Examples
    --------
    >>> sol = Solution()
    >>> sol.vlu = np.ones((10, 5))
    >>> sol.policy.c = np.zeros((10, 5))
    >>> sol["Q"] = np.ones((10, 5))
    >>> sol.EGM.refined.m = np.linspace(0, 1, 20)
    >>> sol.lambda_ = np.ones((10, 5))  # Note: lambda_ not lambda
    """
    
    def __init__(self):
        """Initialize empty Solution container."""
        # Create the numba-compatible internal structure
        self._jit = SimpleNamespace()
        
        # Initialize typed dictionaries for policy and timing
        str_type = types.unicode_type
        array_type = types.float64[:]
        self._jit.policy = typed.Dict.empty(str_type, array_type)
        self._jit.timing = typed.Dict.empty(str_type, types.float64)
        
        # Initialize core arrays as empty (0,) arrays
        self._jit.vlu = np.empty((0,), dtype=np.float64)
        self._jit.Q = np.empty((0,), dtype=np.float64)
        self._jit.lambda_ = np.empty((0,), dtype=np.float64)
        self._jit.phi = np.empty((0,), dtype=np.float64)
        
        # Initialize EGM layers
        # Create the type for nested dict
        nested_dict_type = DictType(str_type, array_type)
        self._jit.EGM = typed.Dict.empty(str_type, nested_dict_type)
        for layer in ["unrefined", "refined", "interpolated"]:
            self._jit.EGM[layer] = typed.Dict.empty(str_type, array_type)
        
        # Create attribute views with shape tracking
        self._policy_view = _AttrView(self._jit.policy)
        self._timing_view = _AttrView(self._jit.timing)  # For incremental timing access
        self._EGM_view = SimpleNamespace()
        for layer in ["unrefined", "refined", "interpolated"]:
            setattr(self._EGM_view, layer, _AttrView(self._jit.EGM[layer]))
        
        # Track which fields have been filled
        self._filled = {
            "vlu": False,
            "Q": False,
            "lambda_": False,
            "phi": False
        }
    
    def make_jit_view(self):
        """Return a SimpleNamespace with only numba-legal fields."""
        return self._jit
    
    # Attribute and mapping access
    def __getattr__(self, key):
        if key == "policy":
            return self._policy_view
        elif key == "EGM":
            return self._EGM_view
        elif key == "timing":
            return self._timing_view
        elif key in ["vlu", "Q", "lambda_", "phi"]:
            return getattr(self._jit, key)
        else:
            # Check if it's a custom array
            if hasattr(self._jit, key):
                return getattr(self._jit, key)
            raise AttributeError(f"Solution has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        if key.startswith('_'):
            object.__setattr__(self, key, value)
        elif key in ["vlu", "Q", "lambda_", "phi"]:
            setattr(self._jit, key, value)
            self._filled[key] = True
        elif key == "timing":
            # Allow direct assignment of timing dict
            if isinstance(value, dict):
                self._jit.timing.clear()
                for k, v in value.items():
                    self._jit.timing[k] = float(v)
            else:
                raise ValueError("timing must be a dict")
        else:
            raise AttributeError(f"Cannot set attribute '{key}' on Solution")
    
    def __getitem__(self, key):
        if key == "policy":
            # Return dict with reshaped arrays
            result = {}
            for k, v in self._jit.policy.items():
                if k in self._policy_view._shapes:
                    result[k] = v.reshape(self._policy_view._shapes[k])
                else:
                    result[k] = v
            return result
        elif key == "EGM":
            return {layer: dict(self._jit.EGM[layer]) 
                    for layer in ["unrefined", "refined", "interpolated"]}
        elif key == "timing":
            return dict(self._jit.timing)
        elif key in ["vlu", "Q", "lambda_", "phi"]:
            return getattr(self._jit, key)
        else:
            # Check if it's a custom array
            if hasattr(self._jit, key):
                return getattr(self._jit, key)
            raise KeyError(f"Solution has no key '{key}'")
    
    def __contains__(self, key):
        """Check if a key exists in the Solution."""
        if key in ["vlu", "Q", "lambda_", "phi"]:
            return self._filled.get(key, False)
        elif key == "policy":
            return len(self._jit.policy) > 0
        elif key == "EGM":
            return len(self._jit.EGM) > 0
        elif key == "timing":
            return len(self._jit.timing) > 0
        else:
            # Check if it's a custom array
            return hasattr(self._jit, key)
    
    def __setitem__(self, key, value):
        if key in ["vlu", "Q", "lambda_", "phi"]:
            setattr(self._jit, key, value)
            self._filled[key] = True
        elif key == "policy":
            if isinstance(value, dict):
                self._jit.policy.clear()
                self._policy_view._shapes.clear()
                for k, v in value.items():
                    arr = np.asarray(v, dtype=np.float64)
                    self._policy_view._shapes[k] = arr.shape
                    self._jit.policy[k] = arr.flatten()
            else:
                raise ValueError("policy must be a dict")
        elif key == "timing":
            if isinstance(value, dict):
                self._jit.timing.clear()
                for k, v in value.items():
                    self._jit.timing[k] = float(v)
            else:
                raise ValueError("timing must be a dict")
        else:
            # Allow arbitrary new arrays
            if isinstance(value, np.ndarray):
                setattr(self._jit, key, value)
                # Optionally track in _filled if we extend it
            else:
                raise KeyError(f"Cannot set key '{key}' on Solution")
    
    # Persistence methods
    def save(self, basename):
        """
        Save solution to disk.
        
        Parameters
        ----------
        basename : str
            Base filename (without extension). Will create:
            - {basename}.npz for array data
            - {basename}_meta.json for metadata
        """
        # Collect arrays for npz
        arrays = {}
        
        # Core arrays
        for key in ["vlu", "Q", "lambda_", "phi"]:
            if self._filled[key]:
                arrays[key] = getattr(self._jit, key)
        
        # Check for any other arrays
        for key in dir(self._jit):
            if not key.startswith('_') and key not in ["vlu", "Q", "lambda_", "phi", "policy", "timing", "EGM"]:
                val = getattr(self._jit, key)
                if isinstance(val, np.ndarray):
                    arrays[key] = val
        
        # Policy arrays
        for key, arr in self._jit.policy.items():
            arrays[f"policy_{key}"] = arr
        
        # EGM arrays
        for layer in ["unrefined", "refined", "interpolated"]:
            for key, arr in self._jit.EGM[layer].items():
                arrays[f"EGM_{layer}_{key}"] = arr
        
        # Save arrays
        np.savez_compressed(f"{basename}.npz", **arrays)
        
        # Save metadata
        # Track other arrays
        other_arrays = []
        for key in dir(self._jit):
            if not key.startswith('_') and key not in ["vlu", "Q", "lambda_", "phi", "policy", "timing", "EGM"]:
                val = getattr(self._jit, key)
                if isinstance(val, np.ndarray):
                    other_arrays.append(key)
        
        meta = {
            "filled": self._filled,
            "timing": dict(self._jit.timing),
            "policy_keys": list(self._jit.policy.keys()),
            "policy_shapes": {k: list(v) for k, v in self._policy_view._shapes.items()},
            "EGM_keys": {
                layer: list(self._jit.EGM[layer].keys())
                for layer in ["unrefined", "refined", "interpolated"]
            },
            "EGM_shapes": {
                layer: {k: list(v) for k, v in getattr(self._EGM_view, layer)._shapes.items()}
                for layer in ["unrefined", "refined", "interpolated"]
            },
            "other_arrays": other_arrays
        }
        
        with open(f"{basename}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    
    @classmethod
    def load(cls, basename):
        """
        Load solution from disk.
        
        Parameters
        ----------
        basename : str
            Base filename (without extension)
            
        Returns
        -------
        Solution
            Loaded solution object
        """
        # Create new instance
        sol = cls()
        
        # Load metadata
        with open(f"{basename}_meta.json", "r") as f:
            meta = json.load(f)
        
        # Load arrays
        arrays = np.load(f"{basename}.npz")
        
        # Restore core arrays
        for key in ["vlu", "Q", "lambda_", "phi"]:
            if meta["filled"][key]:
                setattr(sol._jit, key, arrays[key])
                sol._filled[key] = True
        
        # Restore policy
        for key in meta["policy_keys"]:
            sol._jit.policy[key] = arrays[f"policy_{key}"]
            if "policy_shapes" in meta and key in meta["policy_shapes"]:
                sol._policy_view._shapes[key] = tuple(meta["policy_shapes"][key])
        
        # Restore EGM
        for layer in ["unrefined", "refined", "interpolated"]:
            for key in meta["EGM_keys"][layer]:
                sol._jit.EGM[layer][key] = arrays[f"EGM_{layer}_{key}"]
                if "EGM_shapes" in meta and layer in meta["EGM_shapes"] and key in meta["EGM_shapes"][layer]:
                    getattr(sol._EGM_view, layer)._shapes[key] = tuple(meta["EGM_shapes"][layer][key])
        
        # Restore timing
        for key, value in meta["timing"].items():
            sol._jit.timing[key] = value
        
        # Restore any other arrays
        if "other_arrays" in meta:
            for key in meta["other_arrays"]:
                if key in arrays:
                    setattr(sol._jit, key, arrays[key])
        
        return sol
    
    def pkl(self, path):
        """Save solution using pickle (converts to dict first)."""
        # Convert to dict for pickling (typed dicts can't be pickled)
        data = self.as_dict()
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    @classmethod
    def from_pickle(cls, path):
        """Load solution from pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls.from_dict(data)
    
    def as_dict(self):
        """
        Convert to plain Python dict for inspection or JSON serialization.
        
        Returns
        -------
        dict
            Plain dictionary representation
        """
        # Build policy dict with reshaped arrays
        policy_dict = {}
        for k, v in self._jit.policy.items():
            if k in self._policy_view._shapes:
                policy_dict[k] = v.reshape(self._policy_view._shapes[k])
            else:
                policy_dict[k] = v
        
        # Build EGM dict with reshaped arrays
        egm_dict = {}
        for layer in ["unrefined", "refined", "interpolated"]:
            egm_dict[layer] = {}
            layer_view = getattr(self._EGM_view, layer)
            for k, v in self._jit.EGM[layer].items():
                if k in layer_view._shapes:
                    egm_dict[layer][k] = v.reshape(layer_view._shapes[k])
                else:
                    egm_dict[layer][k] = v
        
        result = {
            "timing": dict(self._jit.timing),
            "policy": policy_dict,
            "EGM": egm_dict
        }
        
        # Add core arrays if filled
        for key in ["vlu", "Q", "lambda_", "phi"]:
            if self._filled[key]:
                result[key] = getattr(self._jit, key)
        
        # Add any other arrays
        for key in dir(self._jit):
            if not key.startswith('_') and key not in ["vlu", "Q", "lambda_", "phi", "policy", "timing", "EGM"]:
                val = getattr(self._jit, key)
                if isinstance(val, np.ndarray):
                    result[key] = val
        
        return result
    
    @classmethod
    def from_dict(cls, d):
        """
        Create Solution from plain dict.
        
        Parameters
        ----------
        d : dict
            Dictionary with solution data
            
        Returns
        -------
        Solution
            New solution object
        """
        sol = cls()
        
        # Set core arrays
        for key in ["vlu", "Q", "lambda_", "phi"]:
            if key in d:
                setattr(sol, key, d[key])
        
        # Set policy
        if "policy" in d:
            # Use __setitem__ which handles shape tracking
            sol["policy"] = d["policy"]
        
        # Set timing
        if "timing" in d:
            sol["timing"] = d["timing"]
        
        # Set EGM
        if "EGM" in d:
            for layer in ["unrefined", "refined", "interpolated"]:
                if layer in d["EGM"]:
                    for key, arr in d["EGM"][layer].items():
                        arr = np.asarray(arr, dtype=np.float64)
                        getattr(sol._EGM_view, layer)._shapes[key] = arr.shape
                        sol._jit.EGM[layer][key] = arr.flatten()
        
        # Set any other arrays
        for key, value in d.items():
            if key not in ["vlu", "Q", "lambda_", "phi", "policy", "timing", "EGM"]:
                if isinstance(value, np.ndarray):
                    setattr(sol._jit, key, value)
        
        return sol
    
    # ---- nopython helpers ----
    add_to_policy = staticmethod(njit(lambda d, k, a: d.__setitem__(k, a)))
    add_to_EGM = staticmethod(njit(lambda d, k, a: d.__setitem__(k, a)))

    # ------------------------------------------------------------------
    #  Python pickling interface – make the object pickle/MPI-safe
    # ------------------------------------------------------------------
    def __getstate__(self):
        """
        Return a picklable representation.

        We simply reuse the existing `as_dict()` helper
        (which already converts Numba typed.Dict → plain dict
        and flattens/reshapes arrays).
        """
        return self.as_dict()

    def __setstate__(self, state):
        """
        Restore from the pickled representation.
        """
        tmp = Solution.from_dict(state)
        # copy all internals over
        self.__dict__.update(tmp.__dict__)


__all__ = ["Solution"] 
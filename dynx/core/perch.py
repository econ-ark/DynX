from typing import Any, Dict, List, Optional, Set, Union
import warnings


class Perch:
    """
    Perch in a CircuitCraft circuit.
    
    In CircuitCraft, each perch has three primary attributes:
    - model: Model representation for this perch
    - sol: A callable object or data (solution, previously named 'up')
    - dist: A callable object or data (distribution, previously named 'down')
    
    Each perch can also store additional data items as needed.
    """
    
    def __init__(self, name: str, data_types: Optional[Dict[str, Any]] = None, model: Any = None):
        """
        Initialize a Perch in the circuit.
        
        Parameters
        ----------
        name : str
            The unique identifier for the perch within the circuit.
        data_types : Dict[str, Any], optional
            Dictionary defining data slots with optional initial values.
            By convention, should include 'sol' and 'dist' keys.
        model : Any, optional
            Model representation for this perch. Can be any type.
        
        Examples
        --------
        >>> perch = Perch("policy_perch", {"sol": None, "dist": None})
        >>> perch = Perch("initial_perch", {"sol": initial_policy, "dist": initial_distribution})
        >>> perch = Perch("modeled_perch", {"sol": None, "dist": None}, model=my_perch_model)
        """
        self.name = name
        
        # Initialize data dictionary with new keys but maintain old ones for compatibility
        if data_types is None:
            self.data = {"sol": None, "dist": None, "up": None, "down": None}
        else:
            self.data = data_types.copy()
            
            # Handle legacy keys if present
            if "up" in self.data and "sol" not in self.data:
                self.data["sol"] = self.data["up"]
            if "down" in self.data and "dist" not in self.data:
                self.data["dist"] = self.data["down"]
                
            # Ensure the perch has sol and dist keys
            if "sol" not in self.data:
                self.data["sol"] = None
            if "dist" not in self.data:
                self.data["dist"] = None
                
            # Keep up/down for backward compatibility
            if "up" not in self.data:
                self.data["up"] = self.data["sol"]
            if "down" not in self.data:
                self.data["down"] = self.data["dist"]
                
        self.model = model
        self._initialized_keys = {k for k, v in self.data.items() if v is not None}
    
    @property
    def has_model(self) -> bool:
        """Check if the perch has a model defined."""
        return self.model is not None
    
    def set_model(self, model: Any) -> None:
        """
        Set the model representation for this perch.
        
        Parameters
        ----------
        model : Any
            Model representation for this perch.
        """
        self.model = model
    
    @property
    def sol(self) -> Any:
        """Get the sol attribute of the perch (solution, previously named 'up')."""
        return self.data.get("sol")
    
    @sol.setter
    def sol(self, value: Any) -> None:
        """Set the sol attribute of the perch (solution, previously named 'up')."""
        self.data["sol"] = value
        self.data["up"] = value  # Keep up in sync for backward compatibility
        self._initialized_keys.add("sol")
        self._initialized_keys.add("up")
    
    @property
    def dist(self) -> Any:
        """Get the dist attribute of the perch (distribution, previously named 'down')."""
        return self.data.get("dist")
    
    @dist.setter
    def dist(self, value: Any) -> None:
        """Set the dist attribute of the perch (distribution, previously named 'down')."""
        self.data["dist"] = value
        self.data["down"] = value  # Keep down in sync for backward compatibility
        self._initialized_keys.add("dist")
        self._initialized_keys.add("down")
    
    # Backward compatibility properties with deprecation warnings
    @property
    def up(self) -> Any:
        """Get the up attribute of the perch (deprecated, use sol instead)."""
        warnings.warn("Perch.up is deprecated, use Perch.sol instead", DeprecationWarning, stacklevel=2)
        return self.data.get("up")
    
    @up.setter
    def up(self, value: Any) -> None:
        """Set the up attribute of the perch (deprecated, use sol instead)."""
        warnings.warn("Perch.up is deprecated, use Perch.sol instead", DeprecationWarning, stacklevel=2)
        self.data["up"] = value
        self.data["sol"] = value  # Keep sol in sync
        self._initialized_keys.add("up")
        self._initialized_keys.add("sol")
    
    @property
    def down(self) -> Any:
        """Get the down attribute of the perch (deprecated, use dist instead)."""
        warnings.warn("Perch.down is deprecated, use Perch.dist instead", DeprecationWarning, stacklevel=2)
        return self.data.get("down")
    
    @down.setter
    def down(self, value: Any) -> None:
        """Set the down attribute of the perch (deprecated, use dist instead)."""
        warnings.warn("Perch.down is deprecated, use Perch.dist instead", DeprecationWarning, stacklevel=2)
        self.data["down"] = value
        self.data["dist"] = value  # Keep dist in sync
        self._initialized_keys.add("down")
        self._initialized_keys.add("dist")
    
    def get_data(self, key: str) -> Any:
        """
        Get data stored in the perch by key.
        
        Parameters
        ----------
        key : str
            Key for the data to retrieve.
            
        Returns
        -------
        Any
            The requested data, or None if not present.
            
        Raises
        ------
        KeyError
            If the key doesn't exist in this perch.
        """
        if key not in self.data:
            raise KeyError(f"Key '{key}' not found in perch '{self.name}'")
        return self.data[key]
    
    def set_data(self, key: str, value: Any) -> None:
        """
        Set data in the perch by key.
        
        Parameters
        ----------
        key : str
            Key for the data to set.
        value : Any
            Value to store.
            
        Raises
        ------
        KeyError
            If the key doesn't exist in this perch.
        """
        if key not in self.data:
            raise KeyError(f"Key '{key}' not found in perch '{self.name}'")
        self.data[key] = value
        self._initialized_keys.add(key)
        
        # Keep sol/up and dist/down in sync
        if key == "sol":
            self.data["up"] = value
            self._initialized_keys.add("up")
        elif key == "up":
            self.data["sol"] = value
            self._initialized_keys.add("sol")
        elif key == "dist":
            self.data["down"] = value
            self._initialized_keys.add("down")
        elif key == "down":
            self.data["dist"] = value
            self._initialized_keys.add("dist")
    
    def add_data_key(self, key: str, initial_value: Any = None) -> None:
        """
        Add a new data key to the perch.
        
        Parameters
        ----------
        key : str
            Key for the new data slot.
        initial_value : Any, optional
            Initial value for the data slot, defaults to None.
        """
        if key in self.data:
            raise ValueError(f"Key '{key}' already exists in perch '{self.name}'")
        self.data[key] = initial_value
        if initial_value is not None:
            self._initialized_keys.add(key)
    
    def is_initialized(self, keys: Optional[Union[str, List[str]]] = None) -> bool:
        """
        Check if specified data keys are initialized.
        
        Parameters
        ----------
        keys : str or List[str], optional
            Key or list of keys to check. If None, checks all keys.
            
        Returns
        -------
        bool
            True if all specified keys are initialized (have non-None values).
        """
        if keys is None:
            keys = list(self.data.keys())
        elif isinstance(keys, str):
            keys = [keys]
            
        return all(k in self._initialized_keys for k in keys)
    
    def get_data_keys(self) -> Set[str]:
        """
        Get all data keys defined in this perch.
        
        Returns
        -------
        Set[str]
            Set of all data keys.
        """
        return set(self.data.keys())
    
    def get_initialized_keys(self) -> Set[str]:
        """
        Get keys that have been initialized with values.
        
        Returns
        -------
        Set[str]
            Set of keys that have values.
        """
        return self._initialized_keys.copy()
    
    def clear_data(self, keys: Optional[Union[str, List[str]]] = None) -> None:
        """
        Clear data (set to None) for specified keys.
        
        Parameters
        ----------
        keys : str or List[str], optional
            Key or list of keys to clear. If None, clears all keys.
        """
        if keys is None:
            keys = list(self.data.keys())
        elif isinstance(keys, str):
            keys = [keys]
            
        for key in keys:
            if key in self.data:
                self.data[key] = None
                self._initialized_keys.discard(key)
                
                # Keep sol/up and dist/down in sync
                if key == "sol":
                    self.data["up"] = None
                    self._initialized_keys.discard("up")
                elif key == "up":
                    self.data["sol"] = None
                    self._initialized_keys.discard("sol")
                elif key == "dist":
                    self.data["down"] = None
                    self._initialized_keys.discard("down")
                elif key == "down":
                    self.data["dist"] = None
                    self._initialized_keys.discard("dist")
                
    def __str__(self) -> str:
        """String representation of the perch."""
        initialized = ", ".join(sorted(self._initialized_keys))
        model_str = ", has_model=True" if self.has_model else ""
        return f"Perch({self.name}{model_str}, initialized=[{initialized}])" 
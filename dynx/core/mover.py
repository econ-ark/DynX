"""
Mover in a CircuitCraft circuit.

In CircuitCraft, movers contain functional relationships between perches, with
two key attributes:
- model: Model representation for the mover (mathematical or otherwise)
- comp: Computational callable instantiated from the model
"""

from typing import Any, Dict, List, Optional, Callable, Union


def convert_legacy_model(legacy_model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert legacy model format to the current FunctionalProblem structure.
    
    Parameters
    ----------
    legacy_model : Dict[str, Any]
        The legacy model dictionary
        
    Returns
    -------
    Dict[str, Any]
        Converted model with proper math/num structure
    """
    # Only convert if the model doesn't already have the new structure
    if 'math' in legacy_model and 'num' in legacy_model:
        return legacy_model.copy()
    
    # Convert legacy model format to new structure
    math_dict = {
        'functions': legacy_model.get('functions', {}),
        'constraints': legacy_model.get('constraints', {}),
        'domains': legacy_model.get('domains', {}),
        'shocks': legacy_model.get('shocks', {}),
        'state': legacy_model.get('state', {
            'arvl': {},
            'dcsn': {},
            'cntn': {}
        })
    }
    
    num_dict = {
        'functions': legacy_model.get('numeric', {}),
        'constraints': legacy_model.get('constraints', []) if isinstance(legacy_model.get('constraints'), list) else [],
        'state_space': legacy_model.get('state_space', {
            'arvl': {
                'dimensions': [],
                'grids': {},
                'ranges': {},
                'shape': {}
            },
            'dcsn': {
                'dimensions': [],
                'grids': {},
                'ranges': {},
                'shape': {}
            },
            'cntn': {
                'dimensions': [],
                'grids': {},
                'ranges': {},
                'shape': {}
            }
        })
    }
    
    # Create structured model
    return {
        'math': math_dict,
        'num': num_dict,
        'parameters': legacy_model.get('parameters', {}),
        'settings': legacy_model.get('settings', {}),
        'methods': legacy_model.get('methods', {})
    }


class Mover:
    """
    Mover in a CircuitCraft circuit.
    
    In CircuitCraft, movers represent functional relationships between perches. 
    Each mover contains:
    
    1. model: The representation of the operation (can be equations, FunctionalProblem instances, 
       instructions, grid sizes, etc.)
    2. comp: The computational callable instantiated from the model
    
    The mover direction determines whether it's a backward or forward operation.
    """
    
    def __init__(self, 
                 source_name: str, 
                 target_name: str,
                 edge_type: str = "forward",
                 model: Optional[Any] = None,
                 comp: Optional[Callable] = None,
                 source_keys: Optional[List[str]] = None,
                 target_key: Optional[str] = None,
                 branch_key: Optional[str] = None,
                 stage_name: Optional[str] = None,
                 agg_rule: str = "assign",
                 legacy_conversion: bool = True):
        """
        Initialize a Mover in the circuit.
        
        Parameters
        ----------
        source_name : str
            The name of the source perch.
        target_name : str
            The name of the target perch.
        edge_type : str
            The type of mover: "forward" or "backward".
        model : Any, optional
            Model representation for this mover. Can be any type.
        comp : Callable, optional
            Computational callable instantiated from the model.
        source_keys : List[str], optional
            Keys from the source perch used in the operation.
        target_key : str, optional
            Key in the target perch where the result is stored.
        branch_key : str, optional
            Label used to index multiple incoming objects inside a perch (dict entry)
        agg_rule : str, default="assign"
            Rule for aggregating multiple source values in fan-in scenarios
        legacy_conversion : bool, default=True
            Whether to automatically convert legacy model formats.
        """
        self.source_name = source_name
        self.target_name = target_name
        self.edge_type = edge_type
        
        # Handle model
        if model is None:
            self.model = {}
        else:
            self.set_model(model, legacy_conversion)
        
        self.comp = comp
        self.source_keys = source_keys or []
        self.target_key = target_key
        self.branch_key = branch_key
        self.agg_rule = agg_rule
        self.stage_name = stage_name
    
    @property
    def has_model(self) -> bool:
        """Check if the mover has a model defined."""
        return self.model is not None and bool(self.model)
        
    @property
    def has_comp(self) -> bool:
        """Check if the mover has a comp function instantiated."""
        return self.comp is not None
    
    def set_model(self, model: Any, legacy_conversion: bool = True) -> None:
        """
        Set the model representation for this mover.
        
        Parameters
        ----------
        model : Any
            Model representation containing all information needed
            to define the computational operation.
        legacy_conversion : bool, default=True
            Whether to automatically convert legacy model formats.
        """
        # If model is None or empty, set an empty dict and return
        if model is None or (isinstance(model, dict) and not model):
            self.model = {}
            return
        
        # If it's a FunctionalProblem instance, use it directly
        if hasattr(model, 'math') and hasattr(model, 'num') and hasattr(model, 'parameters'):
            self.model = model
            return
        
        # For dictionary models
        if isinstance(model, dict):
            # Check if model needs conversion
            if legacy_conversion and 'math' not in model and 'num' not in model:
                self.model = convert_legacy_model(model)
            else:
                self.model = model
        else:
            # For non-dict objects, store directly
            self.model = model
        
    def set_comp(self, comp: Callable) -> None:
        """
        Set the computational callable (comp) for this mover.
        
        Parameters
        ----------
        comp : Callable
            The computational function that will transform data.
        """
        self.comp = comp
        
    def create_comp_from_map(self, comp_factory: Callable[[Any], Callable]) -> None:
        """
        Create a comp function from the mover's model using an external factory function.
        
        Parameters
        ----------
        comp_factory : Callable
            Function that takes a model and returns a comp callable.
            
        Raises
        ------
        ValueError
            If no model is defined for this mover.
        """
        if not self.has_model:
            raise ValueError("Cannot create comp function: No model defined for this mover")
            
        self.comp = comp_factory(self.model)
        
    def execute(self, data: Any) -> Any:
        """
        Execute the mover's comp function with the provided data.
        
        Parameters
        ----------
        data : Any
            Input data for the comp function. This can be a dictionary, array, or any other data type.
            
        Returns
        -------
        Any
            Result from executing the comp function.
            
        Raises
        ------
        ValueError
            If no comp function is defined for this mover.
        """
        if not self.has_comp:
            raise ValueError("Cannot execute: No comp function defined for this mover")
            
        return self.comp(data)
        
    def __str__(self) -> str:
        """String representation of the mover."""
        edge_dir = "→" if self.edge_type == "forward" else "←"
        status = []
        if self.has_model:
            status.append("modeled")
        if self.has_comp:
            status.append("executable")
        status_str = ", ".join(status) if status else "empty"
        return f"Mover({self.source_name} {edge_dir} {self.target_name}, {status_str})" 
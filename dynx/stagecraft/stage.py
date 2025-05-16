"""
Stage Module for CircuitCraft

This module provides a specialized Stage class that extends CircuitBoard
to implement the CDC (Continuation, Decision, Arrival) pattern commonly 
used in economic models, particularly dynamic programming problems.

A Stage is a special type of Circuit with three standard perches:
- arvl (arrival): representing state variables
- dcsn (decision): representing choice variables
- cntn (continuation): representing value functions

This implementation follows a simplified design focused on the core CDC pattern.
"""

# Import modules directly to avoid circular imports
from dynx.core.circuit_board import CircuitBoard
from dynx.core.perch import Perch
from dynx.core.mover import Mover

import numpy as np
import json
import pickle
import os
from datetime import datetime
import yaml
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
import networkx as nx

# Define decision type options
DECISION_TYPES = ["continuous", "discrete"]

class Stage(CircuitBoard):
    """
    Stage class for CDC (Continuation, Decision, Arrival) pattern.
    
    A Stage is a specialized CircuitBoard with three standard perches:
    - arvl (arrival): representing state variables
    - dcsn (decision): representing choice variables
    - cntn (continuation): representing value functions
    
    The Stage class is designed to work with separate representations for 
    model initialization (parsing config) and numerical compilation.
    
    Model representations are provided via:
    - init_rep: Module/object for initializing model structure from config (e.g., model_init)
    - num_rep: Module/object for generating numerical components (e.g., gen_num)
    
    Direct access to the model is provided through:
    - stage.model: The main FunctionalProblem instance for the stage
    - stage.model.math: Mathematical definitions (functions, constraints, domains)
    - stage.model.num: Numerical implementations (functions, constraints, state spaces)
    
    Direct access to perches and movers is provided through properties:
    - stage.arvl, stage.dcsn, stage.cntn: Direct access to perches
    - stage.arvl_to_dcsn, stage.dcsn_to_cntn, etc.: Direct access to movers
    - stage.arvl_to_dcsn.model: Direct access to mover models
    
    Attributes
    ----------
    decision_type : str
        Specifies the type of decision handled by the stage ('continuous' or 'discrete').
        This influences the expected structure of `cntn.sol` and the logic
        within certain movers (especially cntn -> dcsn backward).

    Status flags (see `status_flags` dictionary): Track the state of initialization, 
    compilation, solvability, etc., as defined in the development spec.
    """
    
    def __init__(self, name=None, whisperer=None, config_file=None, init_rep=None, num_rep=None, 
                decision_type: str = "continuous", model_mode: str = "in_situ", master_config=None, config=None):
        """
        Initialize a Stage with three standard perches.
        
        The recommended workflow for Stage creation and configuration:
        
        1. Create instance: 
           `stage = Stage(name="MyStage", init_rep=my_init_rep, num_rep=my_num_rep, decision_type='discrete')`
        
        2. Load configuration: `stage.load_config("config.yaml", master_config=master_config)`
           This uses the init_rep to process the configuration file.
           
        3. Build computational model: `stage.build_computational_model()`
           This uses the num_rep to create the operational implementation.
        
        Parameters
        ----------
        name : str, optional
            Name of the stage
        whisperer : object, optional
            Whisperer object for external solving or OperatorFactory for in_situ mode
        config_file : str, optional
            Path to a configuration file. If provided, loads model structure.
        init_rep : object, optional
            Model representation for initialization (e.g., model_init module)
            This should be the initialize_model function from heptapod_b.api
        num_rep : object, optional
            Model representation for numerical compilation (e.g., gen_num module)
        decision_type : str, optional
            Type of decision: 'continuous' or 'discrete'. Defaults to 'continuous'.
        model_mode : str, optional
            Solver mode: 'in_situ' or 'external'. Defaults to 'in_situ'.
        master_config : dict, optional
            Master configuration for parameter inheritance
        config : dict, optional
            Direct configuration dictionary instead of loading from file
        """
        # Call parent constructor
        super().__init__(name=name or "CDCStage")
        
        # Store model mode
        if model_mode not in ["in_situ", "external"]:
            raise ValueError(f"model_mode must be one of ['in_situ', 'external'], got {model_mode}")
        self.model_mode = model_mode
        
        if model_mode == "in_situ":
            # Store operator factory (formerly whisperer for in-situ mode)
            self.operator_factory = whisperer
        else:
            # Store whisperer for external mode
            self.whisperer = whisperer
        
        # Store model representation modules
        self.init_rep = init_rep
        self.num_rep = num_rep
        
        # Store master config
        self.master_config = master_config
        
        # Store decision type
        if decision_type not in DECISION_TYPES:
            raise ValueError(f"decision_type must be one of {DECISION_TYPES}, got {decision_type}")
        self.decision_type = decision_type
        
        # Add the three standard perches
        self.add_perch(Perch("arvl", data_types={"sol": None, "dist": None}))
        self.add_perch(Perch("dcsn", data_types={"sol": None, "dist": None}))
        # Initialize cntn.sol as a dict to store multiple inputs keyed by branch_key
        self.add_perch(Perch("cntn", data_types={"sol": {}, "dist": None}))
        
        # Add standard forward transitions to the graph - only the essential ones for CDC pattern
        self.forward_graph = nx.DiGraph()
        self.forward_graph.add_edge("arvl", "dcsn")
        self.forward_graph.add_edge("dcsn", "cntn")
        
        # Add standard backward transitions to the graph - only the essential ones for CDC pattern
        self.backward_graph = nx.DiGraph()
        self.backward_graph.add_edge("dcsn", "arvl")
        self.backward_graph.add_edge("cntn", "dcsn")
        
        # Create the forward movers with empty models
        # Forward movers
        self.add_mover(
            source_name="arvl",
            target_name="dcsn",
            edge_type="forward",
            model=None,
            source_keys=["sol", "dist"],
            target_key="dist",
            stage_name=self.name
        )
        
        self.add_mover(
            source_name="dcsn",
            target_name="cntn",
            edge_type="forward",
            model=None,
            source_keys=["sol", "dist"],
            target_key="dist",
            stage_name=self.name
        )
        
        self.add_mover(
            source_name="cntn",
            target_name="arvl",
            edge_type="forward",
            model=None,
            source_keys=["sol", "dist"],
            target_key="dist",
            stage_name=self.name
        )
        
        # Backward movers
        self.add_mover(
            source_name="dcsn",
            target_name="arvl",
            edge_type="backward",
            model=None,
            source_keys=["sol"],
            target_key="sol",
            stage_name=self.name
        )
        
        self.add_mover(
            source_name="cntn",
            target_name="dcsn",
            edge_type="backward",
            model=None,
            source_keys=["sol"],
            target_key="sol",
            stage_name=self.name
        )
        
        # Initialize model attribute with None
        self.model = None
        
        # Add methods attribute to match FunctionalProblem structure
        self.methods = {}
        
        # Internal storage for model representations
        self.__model_representations = {}
        
        # Lifecycle flags -> Status flags
        self.status_flags = {
            "initialized": False,
            "compiled": False,
            "solvable": False,  # Placeholder, logic needs update
            "solved": False,    # Placeholder, logic needs update
            "simulated": False, # Placeholder, logic needs update
            "portable": False,   # Set based on whisperer/packaging, not config/init
            "all_perches_initialized": False, # Placeholder, logic needs update
            "all_perches_compiled": False,   # Placeholder, logic needs update
            "all_movers_initialized": False, # Placeholder, logic needs update
            "all_movers_compiled": False    # Placeholder, logic needs update
        }
        
        # If both a config and init_rep are provided, initialize directly
        if config is not None and self.init_rep is not None:
            # Call the initialize_model function directly with config and master_config
            stage_model, mover_models, perch_models = self.init_rep(config, master_config)
            
            # Store models in model_representations dictionary
            model_representations = {
                'stage_model': stage_model,
                'mover_models': mover_models,
                'perch_models': perch_models
            }
            
            # Store the model representations internally
            self.__model_representations = model_representations
            
            # Extract stage model and configuration
            self.model = stage_model
            
            # Extract perch models and attach to perches
            for perch_name, perch_model in perch_models.items():
                if perch_name in self.perches and perch_model is not None:
                    self.perches[perch_name].model = perch_model
            
            # Set up movers based on the new model representations
            self.setup_movers()
            
            # Set initialized flag
            self.status_flags["initialized"] = True
            
            # Check if all perches are initialized
            all_perches_init = all(p.model is not None for p in self.perches.values())
            self.status_flags["all_perches_initialized"] = all_perches_init
            
            # Check if all movers are initialized
            standard_movers = [
                self._get_mover("arvl", "dcsn", "forward"),
                self._get_mover("dcsn", "cntn", "forward"),
                self._get_mover("cntn", "arvl", "forward"),
                self._get_mover("dcsn", "arvl", "backward"),
                self._get_mover("cntn", "dcsn", "backward")
            ]
            all_movers_init = all(m is not None and m.model is not None for m in standard_movers if m is not None)
            self.status_flags["all_movers_initialized"] = all_movers_init
        
        # Load configuration from file if provided
        elif config_file and self.init_rep:
            self.load_config(config_file, master_config=self.master_config)
    
    def _get_mover(self, source, target, graph_type="forward"):
        """
        Helper method to get a mover from a graph.
        
        Parameters
        ----------
        source : str
            Source perch name
        target : str
            Target perch name
        graph_type : str, optional
            Type of graph: "forward" or "backward"
            
        Returns
        -------
        Mover or None
            The mover if it exists, None otherwise
        """
        graph = self.forward_graph if graph_type == "forward" else self.backward_graph
        if graph.has_edge(source, target) and 'mover' in graph[source][target]:
            return graph[source][target]['mover']
        return None
    
    def setup_movers(self):
        """
        Set up movers based on model representations.
        
        This method checks for the existence of model representations and attaches
        model objects to the existing movers. It does not create new movers, as the
        standard movers (arvl_to_dcsn, dcsn_to_cntn, cntn_to_dcsn, dcsn_to_arvl)
        are already created during initialization.
        
        Returns
        -------
        bool
            True if movers were set up, False otherwise
        """
        # Verify that model representations exist
        if not self.__model_representations:
            return False
            
        # Get mover models from the representations
        mover_models = self.__model_representations.get('mover_models', {})
        
        # Let the plugin handle creating mover models - we don't create defaults here
        if not mover_models:
            return False
        
        # Process mover models and attach them to existing movers
        for mover_name, mover_model in mover_models.items():
            # Skip if mover model is None
            if mover_model is None:
                continue
                
            # Check if the mover name follows the 'source_to_target' pattern
            if '_to_' in mover_name:
                source_perch, target_perch = mover_name.split('_to_')
                
                # Check if both perches exist
                if source_perch in self.perches and target_perch in self.perches:
                    # Determine which graph to use based on mover name
                    if ((source_perch == 'arvl' and target_perch == 'dcsn') or
                        (source_perch == 'dcsn' and target_perch == 'cntn') or
                        (source_perch == 'cntn' and target_perch == 'arvl')):
                        # Forward movers
                        edge_type = "forward"
                        graph = self.forward_graph
                    else:
                        # Backward movers
                        edge_type = "backward"
                        graph = self.backward_graph
                    
                    try:
                        # Check if the mover already exists in the graph
                        if graph.has_edge(source_perch, target_perch) and 'mover' in graph[source_perch][target_perch]:
                            # Get the existing mover
                            mover = graph[source_perch][target_perch]['mover']
                            
                            # Attach the model to the existing mover
                            mover.set_model(mover_model)
                    except Exception:
                        pass
        
        # Verify that movers have models attached
        movers_with_models = 0
        for source, target, data in self.forward_graph.edges(data=True):
            if 'mover' in data and data['mover'] is not None and data['mover'].has_model:
                movers_with_models += 1
                
        for source, target, data in self.backward_graph.edges(data=True):
            if 'mover' in data and data['mover'] is not None and data['mover'].has_model:
                movers_with_models += 1
        
        return movers_with_models > 0
    
    def attach_whisperer_operators(self):
        """
        Attach operators generated by the whisperer to all movers.
        
        This method is maintained for backward compatibility.
        It calls attach_operatorfactory_operators() if in in_situ mode,
        otherwise it assumes external mode and raises a warning.
        
        Parameters
        ----------
        self : 
        """
        if self.model_mode == "in_situ":
            # For in-situ mode, call the operator factory version
            return self.attach_operatorfactory_operators()
        else:
            # For external mode, print a warning
            print("Warning: Using attach_whisperer_operators() in 'external' mode. "
                  "An external whisperer should be used to solve the model instead.")
            
            if not hasattr(self, 'whisperer') or self.whisperer is None:
                raise ValueError("No whisperer provided for external mode. Please set a whisperer first.")
            
            return None
    
    def attach_operatorfactory_operators(self):
        """
        Attach operators generated by the operator_factory to all movers.
        Used for in-situ mode to compile and attach computational functions.
        
        Parameters
        ----------
        self : 
        """
        if self.model_mode != "in_situ":
            raise ValueError("Cannot use attach_operatorfactory_operators in external mode.")
            
        if not hasattr(self, 'operator_factory') or self.operator_factory is None:
            raise ValueError("No operator_factory provided. Please set an operator_factory first.")
        
        # Get operators from operator_factory
        forward_ops = self.operator_factory(self)["forward"]
        backward_ops = self.operator_factory(self)["backward"]
        
        # Standard mover pairs in a CDC stage
        forward_mover_pairs = [("arvl", "dcsn"), ("dcsn", "cntn")]
        backward_mover_pairs = [("dcsn", "arvl"), ("cntn", "dcsn")]
        
        # Set forward operators
        for source, target in forward_mover_pairs:
            mover_name = f"{source}_to_{target}"
            if mover_name in forward_ops and self.forward_graph.has_edge(source, target):
                self.set_mover_comp(source, target, "forward", forward_ops[mover_name])
        
        # Set backward operators
        for source, target in backward_mover_pairs:
            mover_name = f"{source}_to_{target}"
            
            if mover_name in backward_ops and self.backward_graph.has_edge(source, target):
                self.set_mover_comp(source, target, "backward", backward_ops[mover_name])
        
        # Check if all backward movers have comps to determine portability
        all_backward_comps_set = True
        for u, v, data in self.backward_graph.edges(data=True):
            mover = data.get("mover")
            if mover is not None and not mover.has_comp:
                print(f"Warning: Backward mover {u}->{v} lacks a comp after operator factory attachment.")
                all_backward_comps_set = False
        
        self.status_flags["portable"] = all_backward_comps_set
        if all_backward_comps_set:
            print("Stage is portable with all required backward movers.")
        else:
            print("Some required backward movers are missing.")
    
    def initialize_values(self, initial_values=None, state_shape=None, params=None):
        """
        Initialize perch values, potentially using default logic.
        
        Parameters
        ----------
        initial_values : dict, optional
            Dictionary of initial values for each perch
        state_shape : tuple, optional
            Shape of the state space for generating default values
        params : object, optional
            Parameters for generating default values
        
        Returns
        -------
        bool
            True if successfully solved, False otherwise
            
        Raises
        ------
        ValueError
            If the stage is not initialized or not compiled
        RuntimeError
            If the stage is not solvable
        """
        if not self.status_flags["initialized"]:
            raise ValueError("Stage not initialized. Call load_config() first.")
        
        if not self.status_flags["compiled"]:
            raise ValueError("Stage not compiled. Call build_computational_model() first.")
            
        # Check stage-specific solvability
        self._check_solvability()
        if not self.status_flags["solvable"]:
             raise RuntimeError(
                "Cannot solve: Stage is not solvable (check cntn.sol and arvl.dist data)." 
                f" Flags: {self.status_flags}")

        # Directly call the stage's overridden backward and forward methods
        success = True
        try:
            self.solve_backward()
        except Exception as e:
            print(f"Error during backward solving in Stage: {e}")
            success = False
            
        if success:
            try:
                self.solve_forward()
            except Exception as e:
                print(f"Error during forward solving in Stage: {e}")
                success = False
                
        return success

    def solve_backward(self):
        """Override solve_backward to set the stage-specific 'solved' flag."""
        super().solve_backward() # Call the parent logic
        # Check if the primary target perch (arvl) has 'sol' data after solving
        if self.arvl.sol is not None:
             self.status_flags["solved"] = True
        else:
            # Optionally keep it False or add more sophisticated checks
             self.status_flags["solved"] = False 
             print("Warning: Backward solve completed, but arvl.sol is still None.")

    def solve_forward(self):
        """Override solve_forward to set the stage-specific 'simulated' flag."""
        super().solve_forward() # Call the parent logic
        # Check if the primary target perch (cntn) has 'dist' data after solving
        if self.cntn.dist is not None:
             self.status_flags["simulated"] = True
        else:
            # Optionally keep it False or add more sophisticated checks
             self.status_flags["simulated"] = False 
             print("Warning: Forward solve completed, but cntn.dist is still None.")

    def _check_solvability(self):
        """
        Check if the Stage has the minimal data required to be solved.
        Sets the 'solvable' status flag.
        Based on spec: Requires cntn 'sol' data and arvl 'dist' data.
        """
        # Ensure perches exist
        cntn_perch = self.perches.get("cntn")
        arvl_perch = self.perches.get("arvl")
        
        if cntn_perch is None or arvl_perch is None:
            self.status_flags["solvable"] = False
            return
            
        # Check for required data
        has_cntn_sol = cntn_perch.sol is not None
        has_arvl_dist = arvl_perch.dist is not None
        
        self.status_flags["solvable"] = has_cntn_sol and has_arvl_dist

    def __str__(self):
        """String representation of the stage."""
        forward_movers = len(self.forward_graph.edges()) if hasattr(self, 'forward_graph') else 0
        backward_movers = len(self.backward_graph.edges()) if hasattr(self, 'backward_graph') else 0
        
        return f"Stage '{self.name}' (forward_movers: {forward_movers}, backward_movers: {backward_movers})"

    def load_config(self, config_file: str, master_config: Dict = None) -> None:
        """
        Load Stage configuration from a file using the model representation module.
        
        This method uses the init_rep object to process the configuration file
        and apply the resulting model representations to the Stage.
        
        Parameters
        ----------
        config_file : str
            Path to the configuration file
        master_config : Dict, optional
            Master configuration dictionary for parameter inheritance
        
        Raises
        ------
        FileNotFoundError
            If the configuration file doesn't exist
        ValueError
            If the configuration file has invalid format or no init_rep is provided
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        if self.init_rep is None:
            raise ValueError("No initialization model representation (init_rep) provided.")
        
        # Use master_config from parameter or from instance
        master_config = master_config or self.master_config
        
        try:
            # Check if init_rep has initialize_model function (expected interface)
            if hasattr(self.init_rep, 'initialize_model'):
                # Use the init_rep module to initialize the model structure
                # Pass both config_file and master_config
                stage_model, mover_models, perch_models = self.init_rep.initialize_model(config_file, master_config)
                
                # Store models in model_representations dictionary
                model_representations = {
                    'stage_model': stage_model,
                    'mover_models': mover_models,
                    'perch_models': perch_models
                }
                
                # Also include the config for reference
                if hasattr(self.init_rep, 'load_config'):
                    config = self.init_rep.load_config(config_file)
                    model_representations['stage_config'] = config.get('stage', {})
            else:
                # Fall back to legacy method
                model_representations = self.init_rep.load_config(config_file)
                
            # Store the model representations internally
            self.__model_representations = model_representations
            
            # Extract stage model and configuration
            self.model = model_representations.get('stage_model')
            stage_config = model_representations.get('stage_config', {})
            
            # Set stage name if provided
            if 'name' in stage_config:
                self.name = stage_config['name']
            
            # Extract perch models and attach to perches
            perch_models = model_representations.get('perch_models', {})
            for perch_name, perch_model in perch_models.items():
                if perch_name in self.perches and perch_model is not None:
                    self.perches[perch_name].model = perch_model
            
            # Set up movers based on the new model representations
            self.setup_movers()
            
            # Set initialized flag
            self.status_flags["initialized"] = True
            
            # Check if all perches are initialized
            all_perches_init = all(p.model is not None for p in self.perches.values())
            self.status_flags["all_perches_initialized"] = all_perches_init
            
            # Check if all movers are initialized (check standard movers)
            # Note: This assumes standard CDC movers exist. A more robust check might be needed.
            standard_movers = [
                self._get_mover("arvl", "dcsn", "forward"),
                self._get_mover("dcsn", "cntn", "forward"),
                self._get_mover("cntn", "arvl", "forward"),
                self._get_mover("dcsn", "arvl", "backward"),
                self._get_mover("cntn", "dcsn", "backward")
            ]
            all_movers_init = all(m is not None and m.model is not None for m in standard_movers if m is not None)
            # If no backward movers, they don't count against initialization
            if not self.has_backward_movers:
                standard_movers = [m for m in standard_movers if m is None or m.graph_type == "forward"]
                all_movers_init = all(m is not None and m.model is not None for m in standard_movers if m is not None)
            self.status_flags["all_movers_initialized"] = all_movers_init
            
        except Exception as e:
            # Reraise with more context? Or just log? For now, keep as ValueError.
            # Consider logging the original exception `e`
            raise ValueError(f"Error loading configuration file '{config_file}': {str(e)}")
    
    @property
    def has_backward_movers(self):
        """Check if any backward movers exist."""
        return (self._get_mover("dcsn", "arvl", "backward") is not None or 
                self._get_mover("cntn", "dcsn", "backward") is not None)

    @property
    def forward_movers(self):
        """
        Get all forward movers as a dictionary.
        
        Returns
        -------
        dict
            Dictionary mapping mover names to mover objects
        """
        forward_movers = {}
        mover = self._get_mover("arvl", "dcsn", "forward")
        if mover:
            forward_movers["arvl_to_dcsn"] = mover
            
        mover = self._get_mover("dcsn", "cntn", "forward")
        if mover:
            forward_movers["dcsn_to_cntn"] = mover
            
        return forward_movers
    
    @property
    def backward_movers(self):
        """
        Get all backward movers as a dictionary.
        
        Returns
        -------
        dict
            Dictionary mapping mover names to mover objects
        """
        backward_movers = {}
        mover = self._get_mover("dcsn", "arvl", "backward")
        if mover:
            backward_movers["dcsn_to_arvl"] = mover
            
        mover = self._get_mover("cntn", "dcsn", "backward")
        if mover:
            backward_movers["cntn_to_dcsn"] = mover
            
        return backward_movers

    def build_computational_model(self, factory=None):
        """
        Build the computational model from the model representation.
        
        This method generates numerical models from functional problem instances.
        It follows a two-step approach:
        1. If using model_init.py module, first initialize the models
        2. If using gen_num.py module, compile the numerical components
        
        Parameters
        ----------
        factory : object, optional
            Factory object that can create computational objects
            
        Returns
        -------
        bool
            True if the computational model was successfully built
        """
        if self.num_rep is None:
            raise ValueError("No numerical model representation (num_rep) provided.")
        
        try:
            updated_representations = {}
            
            # Check if we have the num_rep module for numerical compilation
            if hasattr(self.num_rep, 'generate_numerical_model'):
                # Extract models from our representations
                stage_model = self.__model_representations.get('stage_model')
                mover_models = self.__model_representations.get('mover_models', {})
                perch_models = self.__model_representations.get('perch_models', {})
                
                # Generate numerical model for stage
                if stage_model is not None:
                    stage_model = self.num_rep.generate_numerical_model(stage_model)
                
                # Generate numerical models for movers
                compiled_mover_models = {}
                for mover_name, mover in mover_models.items():
                    if mover is not None:
                        # Initialize methods to ensure they're a dictionary 
                        if hasattr(self.num_rep, 'initialize_methods'):
                            self.num_rep.initialize_methods(mover)
                        compiled_mover_models[mover_name] = self.num_rep.generate_numerical_model(mover)
                
                # Generate numerical models for perches
                compiled_perch_models = {}
                for perch_name, perch in perch_models.items():
                    if perch is not None:
                        # Initialize methods to ensure they're a dictionary
                        if hasattr(self.num_rep, 'initialize_methods'):
                            self.num_rep.initialize_methods(perch)
                        compiled_perch_models[perch_name] = self.num_rep.generate_numerical_model(perch)
                
                # Update representations with compiled models
                updated_representations = {
                    'stage_model': stage_model,
                    'mover_models': compiled_mover_models,
                    'perch_models': compiled_perch_models,
                    'stage_config': self.__model_representations.get('stage_config', {})
                }
            elif callable(self.num_rep) and not hasattr(self.num_rep, 'build_computational_model'):
                # Direct function approach - use num_rep as the generator function directly
                # Extract models from our representations
                stage_model = self.__model_representations.get('stage_model')
                mover_models = self.__model_representations.get('mover_models', {})
                perch_models = self.__model_representations.get('perch_models', {})
                
                # Generate numerical model for stage
                if stage_model is not None:
                    stage_model = self.num_rep(stage_model)
                
                # Generate numerical models for movers
                compiled_mover_models = {}
                for mover_name, mover in mover_models.items():
                    if mover is not None:
                        compiled_mover_models[mover_name] = self.num_rep(mover)
                
                # Generate numerical models for perches
                compiled_perch_models = {}
                for perch_name, perch in perch_models.items():
                    if perch is not None:
                        compiled_perch_models[perch_name] = self.num_rep(perch)
                
                # Update representations with compiled models
                updated_representations = {
                    'stage_model': stage_model,
                    'mover_models': compiled_mover_models,
                    'perch_models': compiled_perch_models,
                    'stage_config': self.__model_representations.get('stage_config', {})
                }
            else:
                # Legacy approach - use the num_rep's build_computational_model
                updated_representations = self.num_rep.build_computational_model(
                    self.__model_representations, factory)
            
            # Update the model representations
            self.__model_representations = updated_representations
            
            # Update the stage's model
            self.model = updated_representations.get('stage_model')
            
            # Update perch models with the latest versions from representations
            perch_models = updated_representations.get('perch_models', {})
            for perch_name, perch_model in perch_models.items():
                if perch_name in self.perches and perch_model is not None:
                    self.perches[perch_name].model = perch_model
            
            # Update mover models with the latest versions from representations
            mover_models = updated_representations.get('mover_models', {})
            
            # Update forward movers
            for source, target, data in self.forward_graph.edges(data=True):
                mover_name = f"{source}_to_{target}"
                if 'mover' in data and mover_name in mover_models:
                    # Set the mover model
                    mover_model = mover_models[mover_name]
                    data['mover'].set_model(mover_model)
            
            # Update backward movers
            for source, target, data in self.backward_graph.edges(data=True):
                mover_name = f"{source}_to_{target}"
                if 'mover' in data and mover_name in mover_models:
                    # Set the mover model
                    mover_model = mover_models[mover_name]
                    data['mover'].set_model(mover_model)
            
            # Set compiled flag
            self.status_flags["compiled"] = True
            
            # Check if all perches are compiled
            all_perches_compiled = all(p.model is not None for p in self.perches.values())
            self.status_flags["all_perches_compiled"] = all_perches_compiled
            
            # Check if all movers are compiled (check standard movers)
            # Note: This assumes standard CDC movers exist. A more robust check might be needed.
            standard_movers = [
                self._get_mover("arvl", "dcsn", "forward"),
                self._get_mover("dcsn", "cntn", "forward"),
                self._get_mover("cntn", "arvl", "forward"),
                self._get_mover("dcsn", "arvl", "backward"),
                self._get_mover("cntn", "dcsn", "backward")
            ]
            all_movers_compiled = all(m is not None and m.model is not None for m in standard_movers if m is not None)
            # If no backward movers, they don't count against compilation
            if not self.has_backward_movers:
                standard_movers = [m for m in standard_movers if m is None or m.graph_type == "forward"]
                all_movers_compiled = all(m is not None and m.model is not None for m in standard_movers if m is not None)
            self.status_flags["all_movers_compiled"] = all_movers_compiled
            
            # Attach grid proxies to perches for easy grid access
            self._attach_grid_proxies()
            
            return True
        
        except Exception as e:
            print(f"Error building computational model: {e}")
            return False

    def _update_models_from_representations(self):
        """
        Update perch and mover models from the current model representations.
        
        This method updates the models of perches and movers with the latest
        processed versions from model representations. This ensures that
        any changes made during build_computational_model (like creating
        numerical grids) are reflected in the actual perch and mover objects.
        """
        if not self.__model_representations:
            return False
        
        # Update perch models
        updated_perch_models = self.__model_representations.get('perch_models', {})
        if updated_perch_models:
            # Update each perch's model with the latest version
            for perch_name, updated_model in updated_perch_models.items():
                if perch_name in self.perches and updated_model is not None:
                    self.perches[perch_name].model = updated_model
        
        # Update mover models
        updated_mover_models = self.__model_representations.get('mover_models', {})
        if updated_mover_models:
            # Update forward movers
            for mover_name, mover in self.forward_movers.items():
                if mover_name in updated_mover_models and mover is not None:
                    mover.set_model(updated_mover_models[mover_name])
            
            # Update backward movers
            for mover_name, mover in self.backward_movers.items():
                if mover_name in updated_mover_models and mover is not None:
                    mover.set_model(updated_mover_models[mover_name])
        
        return True

    def get_stage_attributes(self):
        """
        Get attributes specific to the Stage class, filtering out CircuitBoard attributes.
        
        This method provides a clean view of attributes defined in the Stage class
        without the inherited CircuitBoard attributes, making it easier to understand
        the Stage-specific functionality.
        
        Returns
        -------
        dict
            Dictionary of Stage-specific attributes and their values
        """
        # Get all instance attributes
        all_attrs = self.__dict__
        
        # Define CircuitBoard-specific attributes to exclude
        circuit_board_attrs = {
            'name', 'graph', 'perches', 'edges', 'functions',
            '_solved', '_ready', '_graph_ready'
        }
        
        # Filter out CircuitBoard attributes and private attributes
        stage_attrs = {k: v for k, v in all_attrs.items()
                      if k not in circuit_board_attrs and not k.startswith('_')}
        
        return stage_attrs
        
    def __dir__(self):
        """
        Customize the attributes returned by dir() to only show Stage-specific attributes.
        
        This override provides a cleaner namespace when using dir(stage), showing only
        the attributes specific to the Stage class while hiding the inherited CircuitBoard
        attributes that are accessible through stage.board.
        
        Returns
        -------
        list
            List of Stage-specific attribute names
        """
        # Get Stage-specific attributes
        stage_attrs = self.get_stage_attributes()
        
        # Add Stage class methods from both __dict__ and parent classes
        stage_members = set(stage_attrs.keys())
        
        # Include methods defined in the Stage class
        for name in dir(self.__class__):
            # Include public methods and properties defined at the Stage level
            if not name.startswith('_') and name not in stage_members:
                # Check if it's defined in Stage class but not CircuitBoard
                if (hasattr(self.__class__, name) and
                    not hasattr(CircuitBoard, name) or
                    getattr(self.__class__, name) is not getattr(CircuitBoard, name, None)):
                    stage_members.add(name)
                    
        # Add some important public attributes/methods that are needed
        essential_methods = {
            'board', 'name', 'solve', 'is_portable', 
            'init_rep', 'num_rep', # Add new reps
            # Direct access properties for perches
            'arvl', 'dcsn', 'cntn',
            # Direct access properties for movers
            'arvl_to_dcsn', 'dcsn_to_cntn',
            'dcsn_to_arvl', 'cntn_to_dcsn'
        }
        stage_members.update(essential_methods)
        
        # Add special methods (dunder methods)
        for name in dir(self.__class__):
            if name.startswith('__') and name.endswith('__'):
                stage_members.add(name)
        
        # Specific attributes/methods to hide from dir() output
        hide_attributes = {
            'backward_movers', 
            'forward_movers', 
            'has_backward_movers', 
            'initialize_values', 
            'movers_backward_exist',
            'model_rep' # Hide the old attribute if present
        }
        
        # Remove attributes that should be hidden
        stage_members = stage_members - hide_attributes
                
        return sorted(stage_members) 

    #
    # Direct access properties for perches
    #
    
    @property
    def arvl(self):
        """Direct access to the 'arvl' (arrival) perch."""
        return self.perches.get('arvl')
        
    @property
    def dcsn(self):
        """Direct access to the 'dcsn' (decision) perch."""
        return self.perches.get('dcsn')
        
    @property
    def cntn(self):
        """Direct access to the 'cntn' (continuation) perch."""
        return self.perches.get('cntn')
    
    #
    # Direct access properties for forward movers
    #
    
    @property
    def arvl_to_dcsn(self):
        """Get the mover from arvl to dcsn."""
        return self._get_mover("arvl", "dcsn", "forward")
    
    @property
    def dcsn_to_cntn(self):
        """Get the mover from dcsn to cntn."""
        return self._get_mover("dcsn", "cntn", "forward")
    
    @property
    def cntn_to_arvl(self):
        """Get the mover from cntn to arvl."""
        return self._get_mover("cntn", "arvl", "forward")
    
    #
    # Direct access properties for backward movers
    #
    
    @property
    def dcsn_to_arvl(self):
        """Get the mover from dcsn to arvl."""
        return self._get_mover("dcsn", "arvl", "backward")
    
    @property
    def cntn_to_dcsn(self):
        """Get the mover from cntn to dcsn."""
        return self._get_mover("cntn", "dcsn", "backward")

    #
    # Grid access support
    #
    
    class _GridProxy:
        """
        Proxy class that provides access to state space grids for a perch.
        
        This class implements a __getattr__ method that will attempt to access
        grids from the perch's model.num.state_space structure.
        """
        def __init__(self, perch):
            """
            Initialize the grid proxy with a reference to its perch.
            
            Parameters
            ----------
            perch : Perch
                The perch object this grid proxy will access grids from.
            """
            self._perch = perch
        
        def __getattr__(self, var):
            """
            Access a grid by attribute name.
            
            Parameters
            ----------
            var : str
                The name of the grid to access (like 'a', 'm', 'age', etc.)
                
            Returns
            -------
            numpy.ndarray
                The grid array if found
                
            Raises
            ------
            AttributeError
                If the grid cannot be found or the model is not initialized
            """
            # Ensure perch has a model with numerical representation
            if not hasattr(self._perch, 'model') or self._perch.model is None:
                raise AttributeError(f"Perch {self._perch.name} has no model")
            
            if not hasattr(self._perch.model, 'num'):
                raise AttributeError(f"Perch {self._perch.name} model has no numerical representation")
            
            num = self._perch.model.num
            
            # Check if num.state_space exists
            if hasattr(num, 'state_space'):
                # Check if perch name exists in state_space
                if self._perch.name in num.state_space:
                    perch_space = num.state_space[self._perch.name]
                    
                    # Check for 'grids' dictionary in perch space
                    if isinstance(perch_space, dict) and 'grids' in perch_space and var in perch_space['grids']:
                        return perch_space['grids'][var]
                    
                    # Check for the variable in 'mesh' dictionary
                    if isinstance(perch_space, dict) and 'mesh' in perch_space and var in perch_space['mesh']:
                        return perch_space['mesh'][var]
                    
                    # Check for direct attribute access in case perch_space is an object
                    if hasattr(perch_space, var):
                        return getattr(perch_space, var)
                    
                    # Check for direct key access if perch_space is a dict
                    if isinstance(perch_space, dict) and var in perch_space:
                        return perch_space[var]
            
            # Fall back to other access patterns for backward compatibility
            
            # Direct attribute access on state_space
            if hasattr(num, 'state_space') and hasattr(num.state_space, var):
                return getattr(num.state_space, var)
                
            # Direct attribute access on num
            if hasattr(num, var):
                return getattr(num, var)
                
            # Within state_space dictionary
            if hasattr(num, 'state_space') and isinstance(num.state_space, dict) and var in num.state_space:
                return num.state_space[var]
            
            # Not found with any pattern
            raise AttributeError(f"Grid '{var}' not found in {self._perch.name}.model")

    class _MeshProxy:
        """
        Proxy class that provides access to state space mesh grids for a perch.
        
        This class implements a __getattr__ method that will attempt to access
        mesh grids from the perch's model.num.state_space structure.
        """
        def __init__(self, perch):
            """
            Initialize the mesh proxy with a reference to its perch.
            
            Parameters
            ----------
            perch : Perch
                The perch object this mesh proxy will access mesh grids from.
            """
            self._perch = perch
        
        def __getattr__(self, var):
            """
            Access a mesh grid by attribute name.
            
            Parameters
            ----------
            var : str
                The name of the mesh grid to access (like 'a', 'm', 'age', etc.)
                
            Returns
            -------
            numpy.ndarray
                The mesh grid array if found
                
            Raises
            ------
            AttributeError
                If the mesh grid cannot be found or the model is not initialized
            """
            # Ensure perch has a model with numerical representation
            if not hasattr(self._perch, 'model') or self._perch.model is None:
                raise AttributeError(f"Perch {self._perch.name} has no model")
            
            if not hasattr(self._perch.model, 'num'):
                raise AttributeError(f"Perch {self._perch.name} model has no numerical representation")
            
            num = self._perch.model.num
            
            # Check if num.state_space exists
            if hasattr(num, 'state_space'):
                # Check if perch name exists in state_space
                if self._perch.name in num.state_space:
                    perch_space = num.state_space[self._perch.name]
                    
                    # Check for 'mesh' dictionary in perch space
                    if isinstance(perch_space, dict) and 'mesh' in perch_space:
                        if var in perch_space['mesh']:
                            return perch_space['mesh'][var]
                    
                    # Check for direct attribute access to mesh in perch_space
                    if hasattr(perch_space, 'mesh') and hasattr(perch_space.mesh, var):
                        return getattr(perch_space.mesh, var)
            
            # Fall back to other access patterns
            
            # Try grids if mesh not found (for convenience)
            if hasattr(self._perch, 'grid'):
                try:
                    return self._perch.grid.__getattr__(var)
                except AttributeError:
                    pass
            
            # Not found with any pattern
            raise AttributeError(f"Mesh grid '{var}' not found in {self._perch.name}.model")

    def _attach_grid_proxies(self):
        """
        Attach grid and mesh proxies to all perches in the stage.
        
        This method is called after the computational model is built
        to provide direct access to state space grids through perch.grid
        and mesh grids through perch.mesh (if they exist).
        """
        for perch_name, perch in self.perches.items():
            if hasattr(perch, 'model') and perch.model is not None:
                if hasattr(perch.model, 'num') and hasattr(perch.model.num, 'state_space'):
                    # Only attach if the model has state_space
                    perch.grid = self._GridProxy(perch)
                    
                    # Check if this perch has mesh grids before attaching mesh proxy
                    if (perch.name in perch.model.num.state_space and 
                        isinstance(perch.model.num.state_space[perch.name], dict) and
                        'mesh' in perch.model.num.state_space[perch.name]):
                        perch.mesh = self._MeshProxy(perch)

    def print_model_structure(self, attr_path=None, filename=None):
        """
        Print a summary structure of the Stage or a detailed structure of a specific component.
        
        If attr_path is None, prints a curated markdown summary of the Stage object,
        focusing on key model components (Stage, Perch, Mover) and their .math/.num attributes.
        Otherwise, prints the detailed structure of the specified attribute path.
        
        Parameters
        ----------
        attr_path : str, optional
            Dot-separated path to a specific attribute, e.g., 'model.math.functions' 
            or 'arvl_to_dcsn.model'. If None, prints the curated Stage summary.
        filename : str, optional
            If provided, saves the generated structure string to this file before printing and returning it.
            
        Returns
        -------
        str
            String representation of the requested structure.
        """
        
        # --- Helper Functions --- START ---
        def get_attr_by_path(obj, path):
            """Get a nested attribute using a dot-separated path."""
            if path is None: # Should not happen if called from the else block below
                return obj 
            
            attrs = path.split('.')
            current_obj = obj
            for attr in attrs:
                # Try getting attribute directly
                if hasattr(current_obj, attr):
                    current_obj = getattr(current_obj, attr)
                # Try getting from dictionary if it's a dict (like perches)
                elif isinstance(current_obj, dict) and attr in current_obj:
                    current_obj = current_obj[attr]
                # Handle graph access specifically if needed (though properties are preferred)
                # Add more specific handling if necessary for complex paths
                else:
                    return f"Attribute '{attr}' not found in path '{path}' starting from {type(obj).__name__}"
            return current_obj

        def get_structure(obj, level=0):
            """Generate structured string representation of an object (basic version)."""
            indent = "  " * level
            result = []
            
            if isinstance(obj, dict):
                if not obj:
                    return f"{indent}(empty dict)"
                
                # Limit display depth/length for large dicts if needed
                items_to_show = list(obj.items())[:10] # Limit items shown for brevity
                
                for key, value in items_to_show:
                    if isinstance(value, (dict, list)) and level < 3: # Limit recursion depth
                        result.append(f"{indent}{key}:")
                        result.append(get_structure(value, level + 1))
                    else:
                        result.append(f"{indent}{key}: {type(value).__name__}") # Show type for deeper/complex values
                if len(obj) > 10:
                     result.append(f"{indent}... ({len(obj) - 10} more items)")

            elif isinstance(obj, list):
                if not obj:
                    return f"{indent}(empty list)"
                
                # Show types or limited items for lists
                if all(not isinstance(x, (dict, list)) for x in obj[:5]): # Show first few simple items type
                     result.append(f"{indent}[{len(obj)} items of type {type(obj[0]).__name__ if obj else 'unknown'}]")
                else: # Recurse for complex lists, limited depth/items
                    result.append(f"{indent}[List with {len(obj)} items]:")
                    for i, item in enumerate(obj[:5]): # Show first 5
                         if isinstance(item, (dict, list)) and level < 3:
                             result.append(f"{indent}  [{i}]:")
                             result.append(get_structure(item, level + 2))
                         else:
                             result.append(f"{indent}  [{i}]: {type(item).__name__}")
                    if len(obj) > 5:
                        result.append(f"{indent}  ... ({len(obj) - 5} more items)")

            elif hasattr(obj, '__dict__') and level < 3: # Basic object structure
                 result.append(f"{indent}{type(obj).__name__}:")
                 # Show limited public attributes
                 attrs_to_show = {k: v for k, v in vars(obj).items() if not k.startswith('_')}[:5]
                 for key, value in attrs_to_show.items():
                     result.append(f"{indent}  {key}: {type(value).__name__}")
                 if len(vars(obj)) > 5:
                      result.append(f"{indent}  ...")

            else: # Fallback for simple types or deep objects
                result.append(f"{indent}{type(obj).__name__}")
            
            return "\n".join(result)
            
        def format_model_summary(model, indent="  "):
            """Helper to format .math and .num components of a model object."""
            summary_lines = []
            if model is None:
                # If the parent object exists but its model is None, indicate that.
                # The calling code handles the case where the parent (e.g., perch) doesn't exist.
                return [f"{indent}model: None"] 
                
            # Helper to format a dictionary attribute (like math or num)
            def format_dict_attr(attr_name, attr_obj, sub_indent="    "):
                lines = []
                # This function should only be called if attr_obj is a non-empty dict
                lines.append(f"{sub_indent}{attr_name}:")
                
                items_added = 0
                items_to_iterate = list(attr_obj.items())
                
                # Iterate and filter out keys with empty collection values
                for k, v in items_to_iterate:
                    # Skip if value is an empty collection
                    if isinstance(v, (dict, list, set)) and not v:
                        continue 
                        
                    # Stop if we've already added 5 non-empty items
                    if items_added >= 5:
                        break
                        
                    lines.append(f"{sub_indent}  - {k}: {type(v).__name__}")
                    items_added += 1
                    
                # Indicate if there were more non-empty items than shown
                # Count total non-empty items first
                total_non_empty = sum(1 for v in attr_obj.values() if not (isinstance(v, (dict, list, set)) and not v))
                
                if total_non_empty > items_added:
                    lines.append(f"{sub_indent}    ... ({total_non_empty - items_added} more non-empty items)")
                elif items_added == 0:
                     # Should not happen if the parent dict was non-empty, but handle just in case
                     lines.append(f"{sub_indent}  (Contains only empty collections)")
                     
                return lines
                
            # Format specific components only if they exist AND are non-empty dictionaries
            components_found = False
            for attr_name in ['parameters', 'settings', 'math', 'num']:
                if hasattr(model, attr_name):
                    attr_value = getattr(model, attr_name, None)
                    # Check if it's a dictionary and not empty
                    if isinstance(attr_value, dict) and attr_value:
                        summary_lines.extend(format_dict_attr(attr_name, attr_value, sub_indent=indent))
                        components_found = True
            
            # If no non-empty dict components were found, indicate the model type
            if not components_found:
                return [f"{indent}model: {type(model).__name__} (No non-empty dict components found)"]
                
            return summary_lines
        # --- Helper Functions --- END ---

        # --- Main Logic ---
        # Determine output filename
        if filename is None:
            # Create a safe filename from the stage name
            safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in self.name)
            output_filename = f"stage_summary_{safe_name}.md"
        else:
            output_filename = filename
            
        if attr_path is None:
            # Generate the curated markdown summary for the Stage
            lines = [f"# Stage Summary: {self.name}", ""]
            
            # --- Stage Model --- 
            lines.append("## Stage Model")
            if self.model:
                lines.extend(format_model_summary(self.model, indent=""))
            else:
                lines.append("(No Stage Model defined)")
            lines.append("") # Spacer

            # --- Perches --- 
            lines.append("## Perches")
            if not self.perches:
                lines.append("(No perches found)")
            else:
                for name, perch in self.perches.items():
                    lines.append(f"### Perch: {name}")
                    lines.append(f"- sol: {type(perch.sol).__name__}")
                    lines.append(f"- dist: {type(perch.dist).__name__}")
                    if perch.model:
                        lines.append(f"- model:")
                        lines.extend(format_model_summary(perch.model, indent="    ")) # Indented summary
                    else:
                         lines.append(f"- model: None")
                    lines.append("") # Spacer

            # --- Movers --- 
            lines.append("## Movers")
            mover_found = False
            # Iterate through both graphs to find movers
            for graph in [self.forward_graph, self.backward_graph]:
                for u, v, data in graph.edges(data=True):
                    mover = data.get('mover')
                    if mover:
                        mover_found = True
                        lines.append(f"### Mover: {mover.source_name}_to_{mover.target_name} ({mover.edge_type})")
                        lines.append(f"- source: {mover.source_name}")
                        lines.append(f"- target: {mover.target_name}")
                        if mover.model:
                            lines.append(f"- model:")
                            lines.extend(format_model_summary(mover.model, indent="    ")) # Indented summary
                        else:
                            lines.append(f"- model: None")
                        lines.append("") # Spacer
            if not mover_found:
                lines.append("(No movers found)")

            result = "\n".join(lines)
            
        else:
            # Original logic for printing detailed structure of a specific attribute path
            target_obj = get_attr_by_path(self, attr_path)
            title = f"Structure of '{attr_path}'"
            
            # Handle cases where the path leads to an error or simple type
            if isinstance(target_obj, str) and ("Attribute" in target_obj and "not found" in target_obj):
                 result = target_obj # Return the error message
            elif not isinstance(target_obj, (dict, list)) and not hasattr(target_obj, '__dict__'):
                 result = f"{title}: {type(target_obj).__name__} ({target_obj})" # Show simple value
            else:
                # Use the recursive helper for complex types
                structure_str = get_structure(target_obj) 
                result = f"{title}:\n{structure_str}"

        # Save the result to the determined filename
        try:
            with open(output_filename, "w", encoding='utf-8') as f:
                f.write(result)
            print(f"Model structure saved to: {output_filename}")
        except IOError as e:
            print(f"Error saving model structure to {output_filename}: {e}")
            
        # Print and return the result regardless of save success
        print(result)
        return result 

    @property
    def _model_representations(self):
        """
        Internal property to access model representations (private implementation detail).
        
        This is intended for internal use only. Public code should use direct model access:
        - stage.model: Access to the stage model
        - stage.arvl.model: Access to perch models
        - stage.arvl_to_dcsn.model: Access to mover models
        
        Returns
        -------
        dict
            The model representations dictionary
        """
        return self.__model_representations
    
    @_model_representations.setter
    def _model_representations(self, value):
        """
        Internal setter for model representations (private implementation detail).
        
        Parameters
        ----------
        value : dict
            The model representations dictionary
        """
        self.__model_representations = value
        
        # Setup other components from the model representations for internal consistency
        if value:
            stage_model = value.get('stage_model')
            if stage_model:
                self.model = stage_model
            
            perch_models = value.get('perch_models', {})
            for perch_name, perch_model in perch_models.items():
                if perch_name in self.perches and perch_model is not None:
                    self.perches[perch_name].model = perch_model
            
            # Set up movers with potentially updated models
            self.setup_movers()

    def create_transpose_connections(self, edge_type: str = "both") -> List[Mover]:
        """
        Create transpose connections for all movers in the stage.
        
        For each forward mover, create a corresponding backward mover, and vice versa.
        This ensures that for any user-defined connection in one direction, 
        there's an automatic corresponding connection in the opposite direction.
        
        Parameters
        ----------
        edge_type : str
            Type of movers to create transposes for: "forward", "backward", or "both".
            Default is "both" (creates both forward and backward transposes).
            
        Returns
        -------
        List[Mover]
            List of newly created transpose movers.
        
        Notes
        -----
        This method uses the parent CircuitBoard's create_transpose_connections method
        to ensure proper data flow conventions:
        - Forward movers: source.cntn.dist -> target.arvl.dist
        - Backward movers: source.arvl.sol -> target.cntn.sol
        """
        created_movers = []
        
        if edge_type in ["forward", "both"]:
            # Create backward transposes for all forward movers
            forward_transposes = super().create_transpose_connections(edge_type="forward")
            created_movers.extend(forward_transposes)
            if forward_transposes:
                print(f"Created {len(forward_transposes)} backward transposes for forward movers in Stage {self.name}")
        
        if edge_type in ["backward", "both"]:
            # Create forward transposes for all backward movers
            backward_transposes = super().create_transpose_connections(edge_type="backward")
            created_movers.extend(backward_transposes)
            if backward_transposes:
                print(f"Created {len(backward_transposes)} forward transposes for backward movers in Stage {self.name}")
        
        return created_movers
    
    # --- Properties for convenient access to perches --- 
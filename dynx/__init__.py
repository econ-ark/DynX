"""
DynX: Dynamic Economic Modeling Framework

DynX is a framework for building and solving dynamic economic models with
a focus on separation of concerns, structured computational workflows,
and powerful dynamic programming tools.
"""

__version__ = "0.1.0"

# Core components
from dynx.core import CircuitBoard, Perch, Mover

# StageCraft components
from dynx.stagecraft import Stage, ModelCircuit, Period
from dynx.stagecraft.config_loader import initialize_model_Circuit

# Runner components 
from dynx.runner import CircuitRunner, RunRecorder, mpi_map

# Public API
__all__ = [
    # Core
    "CircuitBoard", "Perch", "Mover",
    
    # StageCraft
    "Stage", "ModelCircuit", "Period", "initialize_model_Circuit",
    
    # Runner
    "CircuitRunner", "RunRecorder", "mpi_map",
]

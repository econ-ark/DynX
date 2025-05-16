"""
DynX Runner Module

Tools for parameter sweeping and optimization of economic models.

This module provides tools for efficiently running economic models with different 
parameter sets, collecting metrics, and visualizing results. The main components are:

- CircuitRunner: A class for parameter sweeping and visualization
- RunRecorder: A class for collecting metrics during model execution
- mpi_map: Function for parallel execution using MPI
- Samplers: Classes for different parameter sampling strategies
- Plotting helpers: Functions for visualizing results
"""

from dynx.runner.circuit_runner import CircuitRunner, mpi_map, plot_metrics, plot_errors
from dynx.runner.telemetry import RunRecorder
from dynx.runner.sampler import (
    BaseSampler, MVNormSampler, FullGridSampler, 
    LatinHypercubeSampler, FixedSampler, build_design
)

__all__ = [
    'CircuitRunner', 'RunRecorder', 'mpi_map', 
    'plot_metrics', 'plot_errors',
    'BaseSampler', 'MVNormSampler', 'FullGridSampler', 
    'LatinHypercubeSampler', 'FixedSampler', 'build_design'
]

__version__ = "0.1.0"

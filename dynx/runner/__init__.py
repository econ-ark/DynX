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
from dynx.runner.circuit_runner import _write_design_matrix_csv as write_design_matrix_csv
from dynx.runner.telemetry import RunRecorder
from dynx.runner.sampler import (
    BaseSampler, MVNormSampler, FullGridSampler, 
    LatinHypercubeSampler, FixedSampler, build_design
)
from dynx.runner import reference_utils
from dynx.runner import metrics
from dynx.runner.metrics.deviations import (
    dev_c_L2, dev_c_Linf, dev_a_L2, dev_a_Linf,
    dev_v_L2, dev_v_Linf, dev_pol_L2, dev_pol_Linf,
    make_policy_dev_metric
)

__all__ = [
    'CircuitRunner', 'RunRecorder', 'mpi_map', 
    'plot_metrics', 'plot_errors',
    'BaseSampler', 'MVNormSampler', 'FullGridSampler', 
    'LatinHypercubeSampler', 'FixedSampler', 'build_design',
    'reference_utils', 'metrics',
    'dev_c_L2', 'dev_c_Linf', 'dev_a_L2', 'dev_a_Linf',
    'dev_v_L2', 'dev_v_Linf', 'dev_pol_L2', 'dev_pol_Linf',
    'make_policy_dev_metric',
    'write_design_matrix_csv'
]

__version__ = "0.1.0"

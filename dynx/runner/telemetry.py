"""
Telemetry module for dynx_runner.

Provides classes and functions for collecting metrics during model execution.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class RunRecorder:
    """
    Records metrics during model execution.
    
    This class provides a simple interface for solvers and simulators
    to add metrics during model execution, which are then merged with
    metrics from metric functions.
    """
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def add(self, **kv: Any) -> None:
        """
        Add metrics to the recorder.
        
        Args:
            **kv: Key-value pairs of metrics to add.
                  Values can be any picklable Python object.
                  
        Example:
            rec.add(time=3.2, UE_iters=17, custom=obj)
        """
        self.metrics.update(kv) 
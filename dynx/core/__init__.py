"""
Dyn-X core façade.

Re-exports the three central classes so that
`from dynx.core import CircuitBoard, Perch, Mover`
works no matter where the real implementation lives.
"""

from importlib import import_module as _imp

try:
    # -- adjust the import below if your real code lives elsewhere
    _src = _imp("dynx.stagecraft.core")
    CircuitBoard = _src.CircuitBoard
    Perch        = _src.Perch
    Mover        = _src.Mover
except (ModuleNotFoundError, AttributeError) as exc:  # fallback placeholders
    class _Missing:                                    # noqa: D401
        """Placeholder that raises on first use."""
        def __getattr__(self, _):                       # noqa: D401
            raise ImportError(
                "dynx.core couldn't locate the real implementation; "
                "check that dynx.stagecraft is present."
            ) from exc
    CircuitBoard = Perch = Mover = _Missing()           # type: ignore

__all__ = [
    "CircuitBoard",
    "Perch",
    "Mover"
]

"""Logical state and observable helpers.

The implementation lives in ``cat_env.py`` so the black-box simulator has a
single source of truth.  This module provides the explicit challenge-requested
entry point for teammates who want to inspect or reuse just the logical layer.
"""

from cat_env import (
    coherent_state,
    logical_storage_state,
    normalized_superposition,
    storage_observables,
)

__all__ = [
    "coherent_state",
    "logical_storage_state",
    "normalized_superposition",
    "storage_observables",
]

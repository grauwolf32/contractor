"""Tracked research control plane for Contractor.

The package owns declarative research records and validation. It must not be
imported by :mod:`contractor`; production code stays independent of experiment
IDs and decision thresholds.
"""

from research.models import Decision, Experiment, Hypothesis
from research.registry import ResearchRegistry, load_registry

__all__ = ["Decision", "Experiment", "Hypothesis", "ResearchRegistry", "load_registry"]

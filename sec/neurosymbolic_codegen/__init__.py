"""
Neuro-Symbolic Code Generation AI System

A production-grade code generation AI combining neural networks with symbolic reasoning,
multi-agent coordination, adaptive RL, and self-repair mechanisms.
"""

__version__ = "0.1.0"
__author__ = "AI Research Team"
__license__ = "Apache-2.0"

from neurosymbolic_codegen.core import (
    ASTEncoder,
    SMTConnector,
    RewardModel,
    SelfRepairMechanism,
)
from neurosymbolic_codegen.orchestration import NeuroSymbolicOrchestrator
from neurosymbolic_codegen.rag import RAGSubsystem
from neurosymbolic_codegen.security import SecurityAnalyzer

__all__ = [
    "ASTEncoder",
    "SMTConnector",
    "RewardModel",
    "SelfRepairMechanism",
    "NeuroSymbolicOrchestrator",
    "RAGSubsystem",
    "SecurityAnalyzer",
]

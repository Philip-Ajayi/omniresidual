"""
Backward-compatibility adapter:
Routes legacy NebiusReasoningEngine calls directly into the NVIDIA NIM Reasoning Engine.
"""
from core.nim_client import NIMReasoningEngine

class NebiusReasoningEngine(NIMReasoningEngine):
    """Drop-in subclass ensuring full backward compatibility with tests and demo runners."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
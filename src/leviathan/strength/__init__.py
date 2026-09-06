"""Strength-first frozen-weight research. ARC-AGI grid reasoning is a separate
measurement from ARC-Easy science questions. Optional PyTorch is imported lazily.
"""
from .contracts import ArcTask, Example, SearchConfig
from .programs import Program, parse

__all__ = ['ArcTask', 'Example', 'SearchConfig', 'Program', 'parse']

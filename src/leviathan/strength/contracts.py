"""ARC task boundary: demonstrations and query inputs, never query labels.

All task data are immutable tuples. This is an API boundary, not an OS sandbox.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from hashlib import sha256
import json
import math
import time
from typing import Any

Grid = tuple[tuple[int, ...], ...]

def digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def grid(value) -> Grid:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 30:
        raise ValueError('A grid needs 1..30 rows')
    if not isinstance(value[0], (list, tuple)) or not 1 <= len(value[0]) <= 30:
        raise ValueError('A grid needs 1..30 columns')
    width = len(value[0])
    if any(not isinstance(r, (list, tuple)) or len(r) != width for r in value):
        raise ValueError('Non-rectangular grid')
    if any(type(x) is not int or not 0 <= x <= 9 for r in value for x in r):
        raise ValueError('Grid symbols must be integers 0..9, not booleans')
    return tuple(tuple(r) for r in value)

@dataclass(frozen=True)
class Example:
    input: Grid
    output: Grid
    def __post_init__(self):
        object.__setattr__(self, 'input', grid(self.input))
        object.__setattr__(self, 'output', grid(self.output))

@dataclass(frozen=True)
class ArcTask:
    id: str
    examples: tuple[Example, ...]
    queries: tuple[Grid, ...]
    split: str = 'synthetic'
    def __post_init__(self):
        if not self.id or not self.examples or not self.queries:
            raise ValueError('Task ID, demonstrations and query inputs are required')
        if not all(isinstance(x, Example) for x in self.examples):
            raise TypeError('Typed examples required')
        object.__setattr__(self, 'examples', tuple(self.examples))
        object.__setattr__(self, 'queries', tuple(grid(x) for x in self.queries))
    @classmethod
    def from_public(cls, identifier: str, obj: dict, split='synthetic'):
        if not isinstance(obj, dict) or set(obj) != {'train', 'test'}:
            raise ValueError('Only train/test task views accepted')
        if any(set(r) != {'input'} for r in obj['test']):
            raise ValueError('Query outputs must be removed by the evaluator BEFORE solver entry')
        if any(set(r) != {'input', 'output'} for r in obj['train']):
            raise ValueError('Malformed demonstration')
        return cls(identifier, tuple(Example(r['input'], r['output']) for r in obj['train']),
                   tuple(grid(r['input']) for r in obj['test']), split)
    @property
    def support_hash(self):
        return digest([(x.input, x.output) for x in self.examples])
    def view(self):
        return {'train': [{'input': x.input, 'output': x.output} for x in self.examples],
                'test': [{'input': x} for x in self.queries]}

@dataclass(frozen=True)
class SearchConfig:
    # Strength first, with finite protection against runaway work, not latency tuning.
    max_depth: int = 3
    beam_per_view: int = 16
    max_candidates: int = 30000
    max_seconds: float = 90.0
    max_solutions: int = 128
    neural_rounds: int = 3
    proposals_per_round: int = 4
    repair_budget: int = 128
    seed: int = 1607
    activation_trials: bool = False   # opt-in until native-model support gates pass
    def __post_init__(self):
        for name in ('max_depth', 'beam_per_view', 'max_candidates', 'max_solutions', 'proposals_per_round', 'repair_budget'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'Positive integer required: {name}')
        if not 1 <= self.max_depth <= 8 or not 0 <= self.neural_rounds <= 8:
            raise ValueError('Depth/round guard exceeded')
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError('Finite wall-time guard required')
    @property
    def fingerprint(self): return digest(asdict(self))

@dataclass
class SearchMeter:
    config: SearchConfig
    started: float = field(default_factory=time.monotonic)
    candidates: int = 0
    neural_calls: int = 0
    duplicate_programs: int = 0
    duplicate_behaviors: int = 0
    def available(self):
        return self.candidates < self.config.max_candidates and time.monotonic() - self.started < self.config.max_seconds
    def snapshot(self):
        return {'candidates': self.candidates, 'neural_calls': self.neural_calls,
                'duplicate_programs': self.duplicate_programs, 'duplicate_behaviors': self.duplicate_behaviors,
                'elapsed_seconds': time.monotonic()-self.started,
                'limit_reached': not self.available()}

@dataclass(frozen=True)
class Witness:
    program_hash: str
    example_index: int
    expected_shape: tuple[int, int]
    actual_shape: tuple[int, int] | None
    first_wrong_cell: tuple[int, int, int, int] | None = None  # row,col,predicted,observed
    error: str = ''
    def as_dict(self): return asdict(self)

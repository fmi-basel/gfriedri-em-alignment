from dataclasses import dataclass
from dataclasses_json import dataclass_json

from typing import Tuple, List, Optional


@dataclass_json
@dataclass
class VolumeConfig:
    """Class for render volume parameters"""
    path: str
    resolution: List[float]
    chunk_size: List[int]
    preshift_bits: int
    minishard_bits: int


@dataclass_json
@dataclass
class DownsampleConfig:
    downsample_factors: Tuple[int]=(1, 1)
    method: str="mean"

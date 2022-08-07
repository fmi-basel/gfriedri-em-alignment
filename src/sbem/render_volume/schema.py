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


@dataclass_json
@dataclass
class SizeHiearchy:
    volume_size: Optional[List[int]]=None
    shard_size: Optional[List[int]]=None
    chunk_size: Optional[List[int]]=None
    grid_shape_in_chunks: Optional[List[int]]=None
    grid_shape_in_shards: Optional[List[int]]=None
    bits_xyz: Optional[List[int]]=None

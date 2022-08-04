from dataclasses import dataclass
from dataclasses_json import dataclass_json

from typing import Tuple, List, Optional


@dataclass_json
@dataclass
class RenderVolumeConfig:
    """Class for render volume parameters"""
    volume_path: str
    sections: List["SectionRecord"]
    xy_coords: np.ndarray
    resolution: List[int]
    chunk_size: List[int]
    preshift_bits: int
    minishard_bits: int
    downsample: bool=False
    downsample_factors: List[int]=[1, 1]
    downsample_method: str="mean"



"""
    :param volume_path: the path for writing the volume.
    :param sections: a list of SectionRecord objects
    :param xy_coords: N*2 array, N equals number of sections.
                      Each row is XY offset of each section, with respect
                      to the first secion.
    :param resolution: resolution in nanometer in X, Y, Z.
    :param chunk_size: Chunk size for saving chunked volume. Each element
                       corresponds to dimension X, Y, Z.
"""

@dataclass_json
@dataclass
class SizeHiearchy:
    volume_size: Optional[List[int, int, int]]=None
    shard_size: Optional[List[int, int, int]]=None
    chunk_size: Optional[List[int, int, int]]=None
    grid_shape_in_chunks: Optional[List[int, int, int]]=None
    grid_shape_in_shards: Optional[List[int, int, int]]=None
    bits_xyz: Optional[List[int, int, int]]=None

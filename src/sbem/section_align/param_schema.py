from dataclasses import dataclass
from dataclasses_json import dataclass_json

from typing import Tuple, List, Optional


@dataclass_json
@dataclass
class LoadSectionsConfig:
    """Class for load section parameters"""
    sbem_experiment: str
    grid_index: int
    start_section: int
    end_section: int
    exclude_sections: Optional[List[int]] = None


@dataclass_json
@dataclass
class AlignSectionsConfig:
    """Class for align section parameters"""
    crop_sizes: List[Tuple[int, int]]
    range_limit: float
    filter_size: int
    downsample: bool = False
    downsample_factors: Tuple[int, int] = (1, 1)

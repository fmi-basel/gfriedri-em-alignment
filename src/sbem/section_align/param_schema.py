from dataclasses import dataclass
from typing import List, Optional, Tuple

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class LoadSectionsConfig:
    """Class for load section parameters"""

    sbem_experiment: str
    grid_index: int
    start_section: Optional[int] = None
    end_section: Optional[int] = None
    section_num_list: Optional[List[int]] = None
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

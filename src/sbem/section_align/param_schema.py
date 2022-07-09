from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class LoadSectionsConfig:
    """Class for load section parameters"""
    sbem_experiment: str
    block: str
    grid_index: int
    start_section: int
    end_section:int


    def __post_init__(self):
        self.grid_index = int(self.grid_index)
        self.start_section = int(self.start_section)
        self.end_section = int(self.end_section)


@dataclass_json
@dataclass
class AlignSectionsConfig:
    """Class for align section parameters"""
    downscale_factor: int
    range_limit: float
    filter_size: int


    def __post_init__(self):
        self.downscale_factor = int(self.downscale_factor)
        self.range_limit = float(self.range_limit)
        self.filter_size = int(self.filter_size)

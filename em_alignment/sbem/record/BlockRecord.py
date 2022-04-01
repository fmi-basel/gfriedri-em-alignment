import json
from logging import Logger
from os import mkdir
from os.path import exists, join

import numpy as np
from sbem.experiment import Experiment
from sbem.record.SectionRecord import SectionRecord
from tqdm import tqdm


class BlockRecord:
    def __init__(self, experiment: Experiment, block_id: str):
        self.logger = Logger("Block Record")
        self.experiment = experiment
        self.block_id = block_id

        self.sections = {}

        if self.experiment is not None:
            self.experiment.register_block(self)

    def register_section(self, section: SectionRecord):
        self.sections[section.section_id] = section

    def get_section(self, section_num: int, tile_grid_num: int = 0):
        id_ = tuple([section_num, tile_grid_num])
        if id_ in self.sections.keys():
            return self.sections[tuple([section_num, tile_grid_num])]
        else:
            return None

    def get_section_range(self):
        return np.array(
            sorted(section.section_num for section in self.sections.values())
        )

    def has_missing_section(self):
        z_diff = np.diff(self.get_section_range())
        return (z_diff > 1).all()

    def get_missing_sections(self):
        z_diff = np.diff(self.get_section_range())
        return self.get_section_range()[:-1][z_diff > 1]

    def save(self, path):
        if exists(path):
            self.logger.warning("Block already exists.")
        else:
            mkdir(path)
            block_dict = {
                "block_id": self.block_id,
                "n_section": len(self.sections),
                "sections": list(self.sections.keys()),
            }
            with open(join(path, "block.json"), "w") as f:
                json.dump(block_dict, f, indent=4)

            for section in self.sections.values():
                section_dir = join(path, section.get_name())
                section.save(section_dir)

    def load(self, path):
        path_ = join(path, "block.json")
        if not exists(path_):
            self.logger.warning(f"Block not found: {path_}")
        else:
            with open(path_) as f:
                block_dict = json.load(f)

            self.block_id = block_dict["block_id"]
            for (section_num, tile_grid_num) in tqdm(
                block_dict["sections"], desc="Loading Sections"
            ):
                section = SectionRecord(self, section_num, tile_grid_num)
                section.load(join(path, section.get_name()))

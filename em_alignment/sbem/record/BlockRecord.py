import json
from logging import Logger
from os import mkdir
from os.path import exists, join

import numpy as np
from sbem.experiment import Experiment
from sbem.record.SectionRecord import SectionRecord
from tqdm import tqdm


class BlockRecord:
    def __init__(self, experiment: Experiment, block_id: str, save_dir: str):
        self.logger = Logger("Block Record")
        self.experiment = experiment
        self.block_id = block_id
        if save_dir is not None:
            self.save_dir = join(save_dir, self.block_id)
        else:
            self.save_dir = None

        self.sections = {}

        if self.experiment is not None:
            self.experiment.register_block(self)

    def register_section(self, section: SectionRecord):
        self.sections[section.section_id] = section

    def get_section(self, section_num: int, tile_grid_num: int = 0):
        id_ = tuple([section_num, tile_grid_num])
        if id_ in self.sections.keys():
            return self.sections[id_]
        else:
            return None

    def get_section_range(self):
        return np.array(
            sorted(section.section_num for section in self.sections.values())
        )

    def has_missing_section(self):
        z_diff = np.diff(self.get_section_range())
        return (z_diff > 1).any()

    def get_missing_sections(self):
        existing_sections = self.get_section_range()
        missing_sections = []
        for i in range(existing_sections[0], existing_sections[-1]):
            if i not in existing_sections:
                missing_sections.append(i)
        return np.array(missing_sections)

    def save(self):
        assert self.save_dir is not None, "Save dir is not set."
        mkdir(self.save_dir)
        block_dict = {
            "block_id": self.block_id,
            "save_dir": self.save_dir,
            "n_section": len(self.sections),
            "sections": list(self.sections.keys()),
        }
        with open(join(self.save_dir, "block.json"), "w") as f:
            json.dump(block_dict, f, indent=4)

        for section in self.sections.values():
            section.save()

    def load(self, path):
        path_ = join(path, "block.json")
        if not exists(path_):
            self.logger.warning(f"Block not found: {path_}")
        else:
            with open(path_) as f:
                block_dict = json.load(f)

            self.block_id = block_dict["block_id"]
            self.save_dir = block_dict["save_dir"]
            for (section_num, tile_grid_num) in tqdm(
                block_dict["sections"], desc="Loading Sections"
            ):
                section = SectionRecord(self, section_num, tile_grid_num, self.save_dir)
                section.load(join(path, section.get_name()))
